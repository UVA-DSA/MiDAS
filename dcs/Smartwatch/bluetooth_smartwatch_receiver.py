# ble_smartwatch_receiver.py
# pip install bleak

import os, csv, time, sys, signal, asyncio, contextlib
from typing import List, Optional
from multiprocessing import Queue
from threading import Event
from bleak import BleakClient, BleakScanner, BleakError

# For deep debugging, uncomment:
# import logging
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("bleak").setLevel(logging.DEBUG)

# Windows: avoid Proactor/WinRT shutdown quirks
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Defaults (you can override per process)
DEFAULT_SERVICE_UUID = "8b2e0001-5b3a-4f93-9f2a-5e0f5e5f0001"
DEFAULT_CHAR_UUID    = "8b2e0002-5b3a-4f93-9f2a-5e0f5e5f0002"

# Tunables
CONNECT_TIMEOUT_S       = 45.0
START_NOTIFY_TIMEOUT_S  = 10.0
STOP_NOTIFY_TIMEOUT_S   = 2.0
DISCONNECT_TIMEOUT_S    = 3.0
WATCHDOG_IDLE_SEC       = 10.0   # calmer idle threshold
RETRY_BACKOFF_SEC       = 5.0

# How long to wait after connect for services to populate
SERVICE_DETECT_TOTAL_S  = 6.0
SERVICE_DETECT_STEP_S   = 0.5

def _is_mac(s: str) -> bool:
    return bool(s and ":" in s and len(s) >= 17)

def _norm_uuid(u: Optional[str]) -> Optional[str]:
    return u.lower() if isinstance(u, str) else None

async def _find_device(target: str, timeout: float = 8.0, service_uuid: Optional[str] = None):
    """
    Prefer name search; if service_uuid provided, use it to filter scanning.
    Accepts name substring (recommended) or MAC (less reliable on Wear due to RPA).
    """
    name_key = (target or "").lower()
    svc_list = [service_uuid] if service_uuid else None

    # Pass 1: service-filtered scan
    try:
        devices = await BleakScanner.discover(timeout=timeout, service_uuids=svc_list)
        for d in devices:
            if name_key and name_key in (d.name or "").lower():
                return d
    except Exception:
        pass

    # Pass 2: general scan
    try:
        devices = await BleakScanner.discover(timeout=timeout)
        for d in devices:
            if name_key and name_key in (d.name or "").lower():
                return d
    except Exception:
        pass

    # Pass 3: MAC
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
                            thread_stop: Event,
                            service_uuid: Optional[str] = None,
                            char_uuid: Optional[str] = None) -> None:
    """
    BLE receiver.
    - server_ip:    device name substring (preferred) OR MAC
    - service_uuid: override default service UUID (optional)
    - char_uuid:    override default characteristic UUID (optional)
    """
    svc_uuid = _norm_uuid(service_uuid) or _norm_uuid(DEFAULT_SERVICE_UUID)
    chr_uuid = _norm_uuid(char_uuid)    or _norm_uuid(DEFAULT_CHAR_UUID)

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

    def _safe_put(q: Queue, item):
        if thread_stop.is_set():
            return
        try:
            q.put_nowait(item)
        except (BrokenPipeError, OSError, ValueError):
            pass
        except Exception:
            pass

    def _have_char(client: BleakClient, char_uuid: str) -> bool:
        try:
            for s in client.services:
                for ch in s.characteristics:
                    if (ch.uuid or "").lower() == char_uuid:
                        props = [p.lower() for p in getattr(ch, "properties", [])]
                        return ("notify" in props) or ("indicate" in props)
        except Exception:
            pass
        return False

    async def _wait_for_char(client: BleakClient, char_uuid: str,
                             total_s: float, step_s: float) -> bool:
        """
        After connect, Windows may populate services a bit later.
        Poll services for up to total_s to detect our char.
        """
        deadline = time.monotonic() + total_s
        while time.monotonic() < deadline:
            if _have_char(client, char_uuid):
                return True
            # Try to poke discovery; ignore deprecation warning for get_services
            with contextlib.suppress(Exception):
                await client.get_services()
            await asyncio.sleep(step_s)
        return _have_char(client, char_uuid)

    async def _run():
        buffer = ""
        last_rx = time.monotonic()

        while not thread_stop.is_set():
            try:
                print(f"[Smartwatch/BLE] Scanning for {server_ip} ...")
                dev = await _find_device(server_ip, timeout=10.0, service_uuid=svc_uuid)
                if not dev:
                    print("[Smartwatch/BLE] Device not found. Retrying in 5s...")
                    await asyncio.sleep(RETRY_BACKOFF_SEC)
                    continue

                print(f"[Smartwatch/BLE] Found: {dev.address} ({dev.name})")

                # Create client with disconnect callback (no deprecation)
                def _on_disc(_c): print("[Smartwatch/BLE] Disconnected.")
                client = BleakClient(dev, timeout=CONNECT_TIMEOUT_S, disconnected_callback=_on_disc)

                try:
                    await asyncio.wait_for(client.connect(), timeout=CONNECT_TIMEOUT_S)

                    # Brief settle
                    await asyncio.sleep(0.3)

                    # Wait for the exact characteristic to become visible
                    ok = await _wait_for_char(client, chr_uuid,
                                              SERVICE_DETECT_TOTAL_S, SERVICE_DETECT_STEP_S)
                    if not ok:
                        # Print what Windows currently sees (for diagnosis)
                        try:
                            print("[Smartwatch/BLE] No NOTIFY char found. Services visible:")
                            for s in client.services:
                                print("  SVC", s.uuid)
                                for ch in s.characteristics:
                                    print("    CHAR", ch.uuid, ch.properties)
                        except Exception:
                            pass
                        print("[Smartwatch/BLE] Our char not present → disconnect and rescan...")
                        with contextlib.suppress(Exception):
                            await asyncio.wait_for(client.disconnect(), timeout=DISCONNECT_TIMEOUT_S)
                        await asyncio.sleep(0.3)
                        continue

                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        flush_n = 0

                        def handle(_uuid, data: bytearray):
                            nonlocal buffer, flush_n, last_rx
                            if thread_stop.is_set():
                                return
                            last_rx = time.monotonic()
                            pc_ms = time.time_ns() // 1_000_000
                            try:
                                buffer += data.decode('utf-8', errors='ignore')
                                while True:
                                    nl = buffer.find('\n')
                                    if nl < 0:
                                        break
                                    line = buffer[:nl].strip()
                                    buffer = buffer[nl+1:]
                                    if not line:
                                        continue
                                    parts = line.split(',')
                                    if len(parts) != 5:
                                        continue
                                    try:
                                        seq = int(parts[0])
                                        sw_epoch_ms = int(float(parts[1]))
                                        x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                                        # sanity clamps against corrupt chunks
                                        if not (-50 < x < 50 and -50 < y < 50 and -50 < z < 50):
                                            continue
                                    except Exception:
                                        continue
                                    row = [sw_epoch_ms, smartwatch_id, 'accelerometer', x, y, z, seq, pc_ms]
                                    writer.writerow(row)
                                    flush_n += 1
                                    if flush_n >= 50:
                                        f.flush(); flush_n = 0
                                    _safe_put(fifo_queue, row)
                            except Exception:
                                pass

                        print(f"[Smartwatch/BLE] Subscribing to {chr_uuid} ...")
                        try:
                            await asyncio.wait_for(client.start_notify(chr_uuid, handle),
                                                   timeout=START_NOTIFY_TIMEOUT_S)
                        except Exception as e:
                            print(f"[Smartwatch/BLE] start_notify failed: {e!r}. Reconnecting...")
                            with contextlib.suppress(Exception):
                                await asyncio.wait_for(client.disconnect(), timeout=DISCONNECT_TIMEOUT_S)
                            await asyncio.sleep(0.3)
                            continue

                        # Connected loop — reconnect if idle too long
                        while not thread_stop.is_set() and client.is_connected:
                            await asyncio.sleep(0.25)
                            if (time.monotonic() - last_rx) > WATCHDOG_IDLE_SEC:
                                print("[Smartwatch/BLE] IMU idle → reconnecting ...")
                                break

                        # Per-connection cleanup
                        if client.is_connected:
                            with contextlib.suppress(Exception):
                                await asyncio.wait_for(client.stop_notify(chr_uuid),
                                                       timeout=STOP_NOTIFY_TIMEOUT_S)

                finally:
                    with contextlib.suppress(Exception):
                        if client.is_connected:
                            await asyncio.wait_for(client.disconnect(), timeout=DISCONNECT_TIMEOUT_S)
                    await asyncio.sleep(0.2)  # let WinRT drain events

            except (BleakError, asyncio.TimeoutError) as e:
                print(f"[Smartwatch/BLE] BLE/Timeout: {e!r}. Reconnecting in {RETRY_BACKOFF_SEC}s...")
                await asyncio.sleep(RETRY_BACKOFF_SEC)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[Smartwatch/BLE] Error: {e!r}. Reconnecting in {RETRY_BACKOFF_SEC}s...")
                await asyncio.sleep(RETRY_BACKOFF_SEC)

    # Manual loop so Ctrl+C cancels cleanly in a subprocess
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_task = loop.create_task(_run())
    try:
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        thread_stop.set()
        print("[Smartwatch/BLE] Ctrl+C → stopping...")
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        with contextlib.suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
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
    TARGET = os.environ.get("SW_TARGET", "Galaxy Watch5")  # name substring or MAC
    SVC    = os.environ.get("SW_SERVICE_UUID", DEFAULT_SERVICE_UUID)
    CHR    = os.environ.get("SW_CHAR_UUID",    DEFAULT_CHAR_UUID)

    from multiprocessing import Queue
    from threading import Event
    q = Queue(); stop_evt = Event()
    base_dir = "./sw_data"

    def _h(sig, frm):
        stop_evt.set()
        print("\n[Smartwatch/BLE] SIGINT. Stopping...")

    signal.signal(signal.SIGINT, _h)

    try:
        receive_smartwatch_data(TARGET, 0, q, base_dir, "right", stop_evt,
                                service_uuid=SVC, char_uuid=CHR)
    finally:
        try:
            stop_evt.set()
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
