# ble_smartwatch_receiver.py
# pip install bleak

import os, csv, time, sys, signal
from typing import List, Optional
from multiprocessing import Queue
from threading import Event
from bleak import BleakClient, BleakScanner, BleakError

# import logging
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("bleak").setLevel(logging.DEBUG)

import sys, asyncio
if sys.platform.startswith("win"):
    # Avoid Proactor/WinRT race on shutdown
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


SERVICE_UUID = "8b2e0001-5b3a-4f93-9f2a-5e0f5e5f0001".lower()
CHAR_UUID    = "8b2e0002-5b3a-4f93-9f2a-5e0f5e5f0002".lower()

# Generic Attribute: Service Changed (if present we subscribe to it)
GATT_SERVICE_UUID          = "00001801-0000-1000-8000-00805f9b34fb"
SERVICE_CHANGED_CHAR_UUID  = "00002a05-0000-1000-8000-00805f9b34fb"

# Tuning
CONNECT_TIMEOUT_S = 45.0
SERVICES_REFRESH_TIMEOUT_S = 5.0
START_NOTIFY_TIMEOUT_S = 10.0
STOP_NOTIFY_TIMEOUT_S  = 2.0
DISCONNECT_TIMEOUT_S   = 3.0
WATCHDOG_IDLE_SEC = 3.0
RETRY_BACKOFF_SEC = 5.0


writing_enabled = True

def _is_mac(s: str) -> bool:
    return s and ":" in s and len(s) >= 17

async def _find_device(target: str, timeout: float = 8.0):
    """Prefer name search with service filter. MACs on Wear can rotate (RPA)."""
    name_key = (target or "").lower()
    try:
        devices = await BleakScanner.discover(timeout=timeout, service_uuids=[SERVICE_UUID])
        for d in devices:
            if name_key and name_key in (d.name or "").lower():
                return d
    except Exception:
        pass

    try:
        devices = await BleakScanner.discover(timeout=timeout)
        for d in devices:
            if name_key and name_key in (d.name or "").lower():
                return d
    except Exception:
        pass

    if _is_mac(target):
        try:
            d = await BleakScanner.find_device_by_address(target, timeout=timeout)
            if d:
                return d
        except Exception:
            pass
    return None



def receive_smartwatch_data(server_ip: str,
                            server_port: int,            # ignored (compat)
                            fifo_queue: Queue,
                            recording_dir: str,
                            smartwatch_id: str,
                            thread_stop: Event) -> None:

    columns: List[str] = [
        'sw_epoch_ms','wrist_position','sensor_type',
        'value_X_Axis','value_Y_Axis','value_Z_Axis',
        'seq_num','pc_epoch_ms'
    ]

    out_dir = os.path.join(recording_dir, f"smartwatch_data/sw_{smartwatch_id}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'sw_data.csv')
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(columns)

    async def _pick_notify_char(client: BleakClient) -> Optional[str]:
        """Prefer exact CHAR_UUID; else any NOTIFY/INDICATE under our SERVICE_UUID; else None."""
        try:
            for svc in client.services:
                for ch in svc.characteristics:
                    cu = (ch.uuid or "").lower()
                    props = [p.lower() for p in getattr(ch, "properties", [])]
                    if cu == CHAR_UUID and ("notify" in props or "indicate" in props):
                        return ch.uuid
        except Exception:
            pass
        try:
            for svc in client.services:
                if (svc.uuid or "").lower() == SERVICE_UUID:
                    for ch in svc.characteristics:
                        props = [p.lower() for p in getattr(ch, "properties", [])]
                        if "notify" in props or "indicate" in props:
                            return ch.uuid
                    break
        except Exception:
            pass
        return None

    async def _refresh_services(client: BleakClient):
        with contextlib.suppress(Exception):
            await asyncio.wait_for(client.get_services(), timeout=SERVICES_REFRESH_TIMEOUT_S)

    import contextlib  # local import to keep top tidy


    def _safe_put(q, item):
        global writing_enabled
        # Avoid pushing after shutdown and swallow pipe/closed queue errors.
        if thread_stop.is_set() or not writing_enabled:
            print("Thread stop: ", thread_stop.is_set(), "Writing enabled: ", writing_enabled)
            return
        try:
            q.put_nowait(item)
            # print(f"[Smartwatch/BLE] Pushed to queue SAFELY: {item}")
        except (BrokenPipeError, OSError, ValueError):
            # BrokenPipe/OSError: OS pipe closed; ValueError: queue closed
            print("[Smartwatch/BLE] Queue closed or pipe broken.")
            pass
        except Exception:
            # Any other transient queue issue — ignore on shutdown path
            print("[Smartwatch/BLE] Transient queue issue.")
            pass


    async def _run():
        buffer = ""
        last_rx = time.monotonic()

        while not thread_stop.is_set():
            try:
                print(f"[Smartwatch/BLE] Scanning for {server_ip} ...")
                dev = await _find_device(server_ip, timeout=10.0)
                if not dev:
                    print("[Smartwatch/BLE] Device not found. Retrying in 5s...")
                    await asyncio.sleep(RETRY_BACKOFF_SEC); continue

                print(f"[Smartwatch/BLE] Found: {dev.address} ({dev.name})")
                client_kwargs = {"timeout": CONNECT_TIMEOUT_S}
                try:
                    client_kwargs["services_cache_mode"] = "uncached"
                except Exception:
                    pass

                client = BleakClient(dev, **client_kwargs)
                try:
                    # Connect with timeout
                    await asyncio.wait_for(client.connect(), timeout=CONNECT_TIMEOUT_S)

                    # Disconnected callback → just logs; loop detects via is_connected
                    client.set_disconnected_callback(lambda _c: print("[Smartwatch/BLE] Disconnected."))

                    # Give Windows a moment and refresh once
                    await asyncio.sleep(1.0)
                    await _refresh_services(client)

                    # Subscribe to GATT Service Changed (if available)
                    gatt_changed_uuid = None
                    try:
                        for svc in client.services:
                            if (svc.uuid or "").lower() == GATT_SERVICE_UUID:
                                for ch in svc.characteristics:
                                    if (ch.uuid or "").lower() == SERVICE_CHANGED_CHAR_UUID:
                                        props = [p.lower() for p in getattr(ch, "properties", [])]
                                        if "indicate" in props or "notify" in props:
                                            gatt_changed_uuid = ch.uuid
                                        break
                                break
                    except Exception:
                        pass

                    service_changed_flag = False
                    if gatt_changed_uuid:
                        async def _on_sc(_uuid, _data: bytearray):
                            nonlocal service_changed_flag
                            print("[Smartwatch/BLE] Service Changed indication received.")
                            service_changed_flag = True
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(
                                client.start_notify(gatt_changed_uuid, _on_sc),
                                timeout=START_NOTIFY_TIMEOUT_S
                            )

                    # Locate a NOTIFY char
                    notify_char = await _pick_notify_char(client)
                    if not notify_char:
                        try:
                            print("[Smartwatch/BLE] No NOTIFY char found. Services visible:")
                            for svc in client.services:
                                print("  SVC", svc.uuid)
                                for ch in svc.characteristics:
                                    print("    CHAR", ch.uuid, ch.properties)
                        except Exception:
                            pass
                        print(f"[Smartwatch/BLE] Retrying in {RETRY_BACKOFF_SEC}s...")
                        await asyncio.sleep(RETRY_BACKOFF_SEC)
                        continue

                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        flush_n = 0

                        def handle(_, data: bytearray):
                            global writing_enabled

                            if thread_stop.is_set() or not writing_enabled:
                                # print("[Smartwatch/BLE] Thread stop set, exiting..")
                                return
                            nonlocal buffer, flush_n, last_rx
                            pc_ms = time.time_ns() // 1_000_000
                            last_rx = time.monotonic()

                            try:
                                buffer += data.decode('utf-8', errors='replace')
                                *full, buffer = buffer.split('\n')
                                for line in full:

                                    line = line.strip()
                                    if not line: continue
                                    parts = line.split(',')
                                    if len(parts) < 5: continue

                                    pc_ms = time.time_ns() // 1_000_000
                                    seq = int(parts[0])
                                    sw_epoch_ms = int(parts[1])
                                    x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                                    row = [sw_epoch_ms, smartwatch_id, 'accelerometer', x, y, z, seq, pc_ms]
                                    writer.writerow(row)
                                    flush_n += 1
                                    # print(f"[Smartwatch/BLE] Received: {row}")
                                    if flush_n >= 50:
                                        f.flush(); flush_n = 0
                                        
                                    if thread_stop.is_set() or not writing_enabled: 
                                        print("[Smartwatch/BLE] Thread stop set, exiting..")
                                        f.flush()
                                        break
                                    try:
                                        _safe_put(fifo_queue, row)
                                    except Exception as e:
                                        print("[Smartwatch/BLE] Error when writing to FIFO, clearing queue.", e)
                                        with contextlib.suppress(Exception):
                                            while not fifo_queue.empty():
                                                fifo_queue.get_nowait()
                            except Exception:
                                pass

                        print(f"[Smartwatch/BLE] Subscribing to {notify_char} ...")
                        await asyncio.wait_for(
                            client.start_notify(notify_char, handle),
                            timeout=START_NOTIFY_TIMEOUT_S
                        )

                        # Connected loop with re-subscribe logic
                        while not thread_stop.is_set() and client.is_connected:
                            await asyncio.sleep(0.25)

                            # If service changed fired, re-discover + re-subscribe
                            if service_changed_flag:
                                service_changed_flag = False
                                with contextlib.suppress(Exception):
                                    await asyncio.wait_for(
                                        client.stop_notify(notify_char),
                                        timeout=STOP_NOTIFY_TIMEOUT_S
                                    )
                                await _refresh_services(client)
                                # re-pick notify characteristic
                                notify_char = await _pick_notify_char(client) or notify_char
                                print(f"[Smartwatch/BLE] Re-subscribing to {notify_char} after Service Changed ...")
                                try:
                                    await asyncio.wait_for(
                                        client.start_notify(notify_char, handle),
                                        timeout=START_NOTIFY_TIMEOUT_S
                                    )
                                    last_rx = time.monotonic()
                                except Exception as e:
                                    print(f"[Smartwatch/BLE] Re-subscribe failed: {e!r}. Forcing reconnect...")
                                    break

                            # Inactivity watchdog: IMU silent → re-subscribe
                            if (time.monotonic() - last_rx) > WATCHDOG_IDLE_SEC:
                                print("[Smartwatch/BLE] IMU idle → re-subscribing ...")
                                with contextlib.suppress(Exception):
                                    await asyncio.wait_for(
                                        client.stop_notify(notify_char),
                                        timeout=STOP_NOTIFY_TIMEOUT_S
                                    )
                                await _refresh_services(client)
                                notify_char = await _pick_notify_char(client) or notify_char
                                try:
                                    await asyncio.wait_for(
                                        client.start_notify(notify_char, handle),
                                        timeout=START_NOTIFY_TIMEOUT_S
                                    )
                                    last_rx = time.monotonic()
                                except Exception as e:
                                    print(f"[Smartwatch/BLE] Re-subscribe failed: {e!r}. Forcing reconnect...")
                                    break

                        # Cleanup subscriptions for this connection
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(client.stop_notify(notify_char), timeout=STOP_NOTIFY_TIMEOUT_S)
                        with contextlib.suppress(Exception):
                            if gatt_changed_uuid:
                                await asyncio.wait_for(client.stop_notify(gatt_changed_uuid), timeout=STOP_NOTIFY_TIMEOUT_S)

                finally:
                    global writing_enabled
                    writing_enabled = False
                    # Always try to disconnect, but don't hang
                    # inside the `finally:` block for each connection
                    with contextlib.suppress(Exception):
                        # Prevent WinRT from trying to schedule into a loop we’re about to close
                        client.set_disconnected_callback(None)

                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(client.disconnect(), timeout=DISCONNECT_TIMEOUT_S)

                    # Let WinRT fire its internal "session closed" events before we close the loop
                    await asyncio.sleep(0.2)

            except (BleakError, asyncio.TimeoutError) as e:
                print(f"[Smartwatch/BLE] BLE/Timeout: {e!r}. Reconnecting in {RETRY_BACKOFF_SEC}s...")
                await asyncio.sleep(RETRY_BACKOFF_SEC)
            except asyncio.CancelledError:
                # Propagate so outer loop cancellation can cleanly stop the loop
                raise
            except Exception as e:
                print(f"[Smartwatch/BLE] Error: {e!r}. Reconnecting in {RETRY_BACKOFF_SEC}s...")
                await asyncio.sleep(RETRY_BACKOFF_SEC)

    # ---- run with a manual loop so we can cancel everything on Ctrl+C ----
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_task = loop.create_task(_run())
    try:
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        thread_stop.set()
        print("[Smartwatch/BLE] Ctrl+C → stopping...")
        # Cancel all tasks and wait for them
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        with contextlib.suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        # Give Bleak/WinRT handlers a moment to flush
        with contextlib.suppress(Exception):
            loop.run_until_complete(asyncio.sleep(0.2))
    finally:
        with contextlib.suppress(Exception):
            loop.run_until_complete(loop.shutdown_asyncgens())
        with contextlib.suppress(Exception):
            loop.close()


# ---- test harness ----
if __name__ == "__main__":
    print("[Smartwatch/BLE] Running as main program.")
    TARGET = "Galaxy Watch5"   # ← IMPORTANT: use NAME substring, not MAC
    from multiprocessing import Queue
    from threading import Event
    q = Queue(); stop_evt = Event()
    base_dir = "./sw_data"

    def _h(sig, frm):
        stop_evt.set()
        print("\n[Smartwatch/BLE] SIGINT. Stopping...")

    signal.signal(signal.SIGINT, _h)

    try:
        receive_smartwatch_data(TARGET, 0, q, base_dir, "right", stop_evt)
    finally:
        try:
            stop_evt.set()
            # Give the async task a brief moment to unwind
            time.sleep(0.2)

            if hasattr(q, "close"):
                try: q.close()
                except Exception: pass
                if hasattr(q, "join_thread"):
                    try: q.join_thread()
                    except Exception: pass
        except Exception:
            pass
        print("[Smartwatch/BLE] Exited cleanly.")
        # last resort if some WinRT handle still lingers:
        # import os; os._exit(0)