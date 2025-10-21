#!/usr/bin/python2.7

import torch
import numpy as np
import random
import os
import argparse
import torch.nn as nn
from loguru import logger
import sys
import torch.optim as optim
import torch.nn.functional as F


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask




class Trainer:
    def __init__(self, model, num_classes, dataset, split):
        print("Number of classes: ", num_classes)
        self.model = model
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

        logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()

                

                # print("target", batch_target)
                # re-arrange to (B, T, C)
                # batch_input = batch_input.permute(0, 2, 1)
                # print("Input shape:", batch_input.shape)
                logits = self.model(batch_input) # shape (B, num_classes, T)
                # print("Prediction shape:", logits.shape)
                # re-arrange to (B, T, C)
                # logits = logits.permute(0, 2, 1)

                print("target", batch_target[:3])


            # CE identical to the original repo
                loss = self.ce(
                    logits.transpose(2,1).contiguous().view(-1, self.num_classes),
                    batch_target.view(-1)
                )
                # optional temporal smoothing like MS-TCN++:
                loss += 0.15 * torch.mean(
                    torch.clamp(
                        self.mse(F.log_softmax(logits[:, :, 1:], dim=1),
                            F.log_softmax(logits.detach()[:, :, :-1], dim=1)),
                        0, 16
                    ) * mask[:, :, 1:]
                )

                # for p in predictions:
                #     loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                #     loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                predicted = torch.argmax(logits, dim=1)

                print("predicted", predicted[:3])
                
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            # torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            logger.info("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                #print vid
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                # re-arrange to (B, T, C)
                input_x = input_x.permute(0, 2, 1)
                print("Input shape:", input_x.shape)
                prediction = self.model(input_x)
                print("Prediction shape:", prediction.shape)
                prediction = prediction.unsqueeze(2).repeat(1, 1, features.shape[1]) # shape (B, num_classes, T)
                predicted = torch.argmax(prediction, dim=1)
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

    def sanity_check(self, model, batch_gen, device, num_classes=None, ignore_index=-100, steps=300):
        """
        Run CE-only overfit on a single batch with extensive diagnostics.
        If model can't overfit, also tries a 1x1 Conv baseline head on the raw input.
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        torch.set_grad_enabled(True)
        model.train().to(device)

        # ---- Fetch one batch ----
        batch_input, batch_target, mask = batch_gen.next_batch(1)
        x = batch_input.to(device)        # expected (B, C_in, T)
        y = batch_target.to(device)       # expected (B, T)
        m = mask.to(device)               # expected (B, 1, T)

        # Fallback to all-ones mask for this test
        if m.dim() == 2: m = m.unsqueeze(1)
        m = torch.ones((x.size(0), 1, x.size(2)), device=device, dtype=torch.float32)

        # ---- One dry forward to inspect shapes ----
        logits = model(x)
        print("Model output shape:", tuple(logits.shape) if isinstance(logits, torch.Tensor) else "N/A")
        # Normalize to (B,C,T) if model returns multi-stage or list

        B, C, T = logits.shape
        if num_classes is None:
            num_classes = C  # infer

        print("=== Sanity Check: batch ===")
        print(f"x shape {tuple(x.shape)}, dtype {x.dtype}")
        print(f"y shape {tuple(y.shape)}, dtype {y.dtype}")
        print(f"mask shape {tuple(m.shape)}, dtype {m.dtype}")
        print(f"logits (dry) shape {tuple(logits.shape)}  (C inferred = {C})")

        # ---- Basic assertions ----
        assert x.dim() == 3, f"Expected x (B,C_in,T), got {tuple(x.shape)}"
        assert y.dim() == 2, f"Expected y (B,T), got {tuple(y.shape)}"
        assert m.shape == (B, 1, T), f"mask must be (B,1,T), got {tuple(m.shape)}"
        assert logits.shape == (B, C, T), f"logits must be (B,C,T), got {tuple(logits.shape)}"
        assert y.dtype == torch.long, f"y must be torch.long, got {y.dtype}"

        # ---- Label stats ----
        y_min = int(y[y != ignore_index].min().item()) if (y != ignore_index).any() else ignore_index
        y_max = int(y[y != ignore_index].max().item()) if (y != ignore_index).any() else ignore_index
        ignore_count = int((y == ignore_index).sum().item())
        print(f"labels: min={y_min}, max={y_max}, ignore_count={ignore_count}, num_classes={num_classes}")
        if (y != ignore_index).any() and (y_max >= num_classes or y_min < 0):
            print("!! Label out of range for CE. Fix label mapping or num_classes.")
            return {"ok": False, "reason": "label_range"}

        # ---- CE-only overfit on one batch ----
        ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

        def _print_diag(step, logits_cur):
            with torch.no_grad():
                lm = logits_cur.mean().item()
                ls = logits_cur.std().item()
                pred = logits_cur.argmax(dim=1)                  # (B,T)
                valid = (y != ignore_index)
                pred_hist = torch.bincount(pred[valid].flatten(), minlength=num_classes).cpu().tolist()
                print(f"step {step:4d} | logits mean/std: {lm:.4f}/{ls:.4f} | pred_hist[:10]: {pred_hist[:10]}")

        print("=== CE-only overfit on 1 batch ===")
        loss_trace = []
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            out = model(x)

            # shape check before loss
            assert out.shape == (B, C, T), f"model output changed shape to {tuple(out.shape)}"
            if torch.isnan(out).any() or torch.isinf(out).any():
                print("!! NaN/Inf detected in logits")
                return {"ok": False, "reason": "nan_in_logits"}

            loss = ce(out.transpose(2, 1).reshape(-1, C), y.reshape(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                print("!! NaN/Inf detected in CE loss")
                return {"ok": False, "reason": "nan_in_loss"}

            loss.backward()

            # grad diagnostics
            with torch.no_grad():
                grad_means = []
                nz = 0
                for p in model.parameters():
                    if p.grad is not None:
                        gm = p.grad.abs().mean().item()
                        grad_means.append(gm)
                        nz += int((p.grad.abs() > 0).sum().item())
                if len(grad_means) == 0 or sum(grad_means) == 0.0:
                    print("!! No gradients flowing (all zeros). Check frozen layers/permutes.")
                    return {"ok": False, "reason": "no_gradients"}

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            loss_trace.append(float(loss.item()))
            if step % 25 == 0:
                print(f"CE loss: {loss.item():.4f}")
                _print_diag(step, out)

        print(f"final CE-only loss: {loss_trace[-1]:.4f}  (start: {loss_trace[0]:.4f})")

        # Success criterion: loss should drop substantially on a single batch
        if loss_trace[-1] < max(0.2, 0.1 * loss_trace[0]):
            print("✅ Model can overfit a single batch with CE-only.")
            return {"ok": True, "mode": "model_ce_overfit", "start_loss": loss_trace[0], "end_loss": loss_trace[-1]}

        print("⚠️  Model did not overfit. Trying 1x1 Conv baseline head on raw input to isolate data/labels...")

        # ---- 1x1 Conv baseline head on raw input (no model), to isolate labels/data ----
        head = nn.Conv1d(in_channels=x.size(1), out_channels=num_classes, kernel_size=1).to(device)
        opt2 = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=0)
        ce2 = nn.CrossEntropyLoss(ignore_index=ignore_index)

        base_trace = []
        for step in range(min(steps, 400)):
            opt2.zero_grad(set_to_none=True)
            logits_base = head(x)  # (B,num_classes,T)
            loss_base = ce2(logits_base.transpose(2, 1).reshape(-1, num_classes), y.reshape(-1))
            loss_base.backward()
            opt2.step()
            base_trace.append(float(loss_base.item()))
            if step % 50 == 0:
                with torch.no_grad():
                    lm = logits_base.mean().item(); ls = logits_base.std().item()
                    pred = logits_base.argmax(1)
                    valid = (y != ignore_index)
                    ph = torch.bincount(pred[valid].flatten(), minlength=num_classes).cpu().tolist()
                    print(f"[baseline] step {step:4d} | loss {loss_base.item():.4f} | logits mean/std {lm:.4f}/{ls:.4f} | pred_hist[:10]: {ph[:10]}")

        print(f"[baseline] final loss: {base_trace[-1]:.4f}  (start: {base_trace[0]:.4f})")

        if base_trace[-1] < max(0.2, 0.1 * base_trace[0]):
            print("✅ Baseline 1x1 head can overfit: issue is likely model architecture/normalization/permutes.")
            return {"ok": False, "reason": "model_arch", "model_loss_end": loss_trace[-1], "baseline_loss_end": base_trace[-1]}
        else:
            print("❌ Baseline cannot overfit either: issue is likely labels (range/ignore), dataloader, or target/mask alignment.")
            print("Check: label mapping to [0..C-1], padded labels set to -100, and that T matches.")
            return {"ok": False, "reason": "data_labels", "model_loss_end": loss_trace[-1], "baseline_loss_end": base_trace[-1]}


# test block

if __name__ == "__main__":
    # use the full temporal resolution @ 15fps

    args = argparse.Namespace()
    args.dataset = "50salads"
    args.split = "1"
    args.num_epochs = 50
    args.features_dim = 2048
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps
    if args.dataset == "50salads":
        sample_rate = 2

    rivanna_standard_root = "/standard/UVA-DSA/MSTCN/"
    vid_list_file = rivanna_standard_root +args.dataset+"/splits/train.split"+args.split+".bundle"
    vid_list_file_tst = rivanna_standard_root +args.dataset+"/splits/test.split"+args.split+".bundle"
    features_path = rivanna_standard_root +args.dataset+"/features/"
    gt_path = rivanna_standard_root +args.dataset+"/groundTruth/"

    mapping_file = rivanna_standard_root +args.dataset+"/mapping.txt"

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_classes = len(actions_dict)

    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)

    batch_gen.read_data(vid_list_file)
    batch_input_tensor, batch_target_tensor, mask = batch_gen.next_batch(1)
    print(batch_input_tensor.size())
    print(batch_target_tensor.size())
    print(mask.size())

    