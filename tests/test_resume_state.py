import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import pcb_vlm
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup
from pcb_vlm.data.resumable_sampler import ResumableBatchSampler


class TestResumeState(unittest.TestCase):
    def test_trainer_state_saving_and_restoration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Setup simple toy model and optimizer
            model = nn.Linear(10, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=20)

            # Take 3 dummy steps
            for _ in range(3):
                out = model(torch.randn(4, 10))
                loss = out.sum()
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            expected_lr = scheduler.get_last_lr()[0]
            saved_rng = torch.get_rng_state()

            # Save state
            state_path = os.path.join(tmp_dir, "trainer_state.pt")
            torch.save({
                "step": 3,
                "epoch": 0,
                "batch_in_epoch": 3,
                "total_batches_processed": 12,
                "best_val_loss": 0.456,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": saved_rng,
            }, state_path)

            self.assertTrue(os.path.exists(state_path))

            # 2. Re-create new model, optimizer, scheduler
            new_model = nn.Linear(10, 2)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
            new_scheduler = get_cosine_schedule_with_warmup(new_optimizer, num_warmup_steps=5, num_training_steps=20)

            loaded_state = torch.load(state_path, map_location="cpu")
            self.assertEqual(loaded_state["step"], 3)
            self.assertEqual(loaded_state["epoch"], 0)
            self.assertEqual(loaded_state["batch_in_epoch"], 3)
            self.assertEqual(loaded_state["total_batches_processed"], 12)
            self.assertEqual(loaded_state["best_val_loss"], 0.456)

            new_optimizer.load_state_dict(loaded_state["optimizer_state_dict"])
            new_scheduler.load_state_dict(loaded_state["scheduler_state_dict"])
            torch.set_rng_state(loaded_state["rng_state"])

            # Verify restored scheduler gives identical LR
            self.assertEqual(new_scheduler.get_last_lr()[0], expected_lr)

    def test_resumable_batch_sampler_deterministic_sequence(self):
        """Verify that interrupting and resuming yields the EXACT sample sequence as uninterrupted run."""
        dataset_len = 12
        batch_size = 2
        seed = 42

        # 1. Uninterrupted continuous run across 2 epochs
        continuous_sampler = ResumableBatchSampler(dataset_len, batch_size=batch_size, seed=seed)
        continuous_samples = []

        for ep in range(2):
            continuous_sampler.set_epoch(ep)
            for batch in continuous_sampler:
                continuous_samples.append(batch)

        # Total 12 samples / 2 per batch = 6 batches per epoch * 2 epochs = 12 batches
        self.assertEqual(len(continuous_samples), 12)

        # 2. Interrupted run: process 3 batches in epoch 0, then checkpoint
        sampler_part1 = ResumableBatchSampler(dataset_len, batch_size=batch_size, seed=seed)
        sampler_part1.set_epoch(0)
        part1_samples = []
        for idx, batch in enumerate(sampler_part1):
            part1_samples.append(batch)
            if idx == 2:  # 3 batches (offset 0, 1, 2)
                saved_epoch = 0
                saved_batch_offset = idx + 1  # 3
                saved_perm = sampler_part1.get_permutation()
                break

        self.assertEqual(part1_samples, continuous_samples[:3])

        # 3. Resumed run from checkpoint
        resumed_sampler = ResumableBatchSampler(dataset_len, batch_size=batch_size, seed=seed)
        resumed_samples = list(part1_samples)

        for ep in range(saved_epoch, 2):
            if ep == saved_epoch and saved_batch_offset > 0:
                resumed_sampler.set_epoch(ep, start_batch_idx=saved_batch_offset, permutation=saved_perm)
            else:
                resumed_sampler.set_epoch(ep, start_batch_idx=0)
            for batch in resumed_sampler:
                resumed_samples.append(batch)

        # The sample sequence must be 100% identical to the continuous run!
        self.assertEqual(len(resumed_samples), len(continuous_samples))
        self.assertEqual(resumed_samples, continuous_samples)

    def test_continuous_vs_interrupted_training_exact_weights(self):
        """Verify that training weights and losses are bit-for-bit or numerically identical between continuous and resumed."""
        torch.manual_seed(1234)
        inputs = torch.randn(8, 4)
        targets = torch.randn(8, 2)

        def make_model():
            torch.manual_seed(999)
            return nn.Linear(4, 2)

        # Run 1: Continuous 6 steps (2 epochs, 4 batches per epoch)
        model_cont = make_model()
        opt_cont = torch.optim.AdamW(model_cont.parameters(), lr=1e-2)
        sched_cont = get_cosine_schedule_with_warmup(opt_cont, num_warmup_steps=2, num_training_steps=6)

        sampler_cont = ResumableBatchSampler(dataset_len=8, batch_size=2, seed=42)
        step_cont = 0
        cont_losses = []

        for ep in range(2):
            sampler_cont.set_epoch(ep)
            for batch_indices in sampler_cont:
                b_in = inputs[batch_indices]
                b_tgt = targets[batch_indices]
                out = model_cont(b_in)
                loss = nn.functional.mse_loss(out, b_tgt)
                loss.backward()
                opt_cont.step()
                sched_cont.step()
                opt_cont.zero_grad()
                cont_losses.append(loss.item())
                step_cont += 1
                if step_cont >= 6:
                    break
            if step_cont >= 6:
                break

        # Run 2: Interrupted at step 3, saved, and resumed
        model_res = make_model()
        opt_res = torch.optim.AdamW(model_res.parameters(), lr=1e-2)
        sched_res = get_cosine_schedule_with_warmup(opt_res, num_warmup_steps=2, num_training_steps=6)

        sampler_res = ResumableBatchSampler(dataset_len=8, batch_size=2, seed=42)
        step_res = 0
        res_losses = []

        # Part A: Run until step 3
        saved_checkpoint = {}
        for ep in range(2):
            sampler_res.set_epoch(ep)
            b_in_epoch = 0
            for batch_indices in sampler_res:
                b_in = inputs[batch_indices]
                b_tgt = targets[batch_indices]
                out = model_res(b_in)
                loss = nn.functional.mse_loss(out, b_tgt)
                loss.backward()
                opt_res.step()
                sched_res.step()
                opt_res.zero_grad()
                res_losses.append(loss.item())
                step_res += 1
                b_in_epoch += 1

                if step_res == 3:
                    saved_checkpoint = {
                        "step": step_res,
                        "epoch": ep,
                        "batch_in_epoch": b_in_epoch,
                        "epoch_indices": sampler_res.get_permutation(),
                        "model_state": {k: v.clone() for k, v in model_res.state_dict().items()},
                        "opt_state": opt_res.state_dict(),
                        "sched_state": sched_res.state_dict(),
                        "rng_state": torch.get_rng_state()
                    }
                    break
            if step_res == 3:
                break

        # Part B: Resume from saved_checkpoint
        model_res2 = make_model()
        model_res2.load_state_dict(saved_checkpoint["model_state"])
        opt_res2 = torch.optim.AdamW(model_res2.parameters(), lr=1e-2)
        sched_res2 = get_cosine_schedule_with_warmup(opt_res2, num_warmup_steps=2, num_training_steps=6)
        opt_res2.load_state_dict(saved_checkpoint["opt_state"])
        sched_res2.load_state_dict(saved_checkpoint["sched_state"])
        torch.set_rng_state(saved_checkpoint["rng_state"])

        start_ep = saved_checkpoint["epoch"]
        start_b_offset = saved_checkpoint["batch_in_epoch"]
        saved_perm = saved_checkpoint["epoch_indices"]
        step_res2 = saved_checkpoint["step"]

        for ep in range(start_ep, 2):
            if ep == start_ep and start_b_offset > 0:
                sampler_res.set_epoch(ep, start_batch_idx=start_b_offset, permutation=saved_perm)
            else:
                sampler_res.set_epoch(ep, start_batch_idx=0)

            for batch_indices in sampler_res:
                b_in = inputs[batch_indices]
                b_tgt = targets[batch_indices]
                out = model_res2(b_in)
                loss = nn.functional.mse_loss(out, b_tgt)
                loss.backward()
                opt_res2.step()
                sched_res2.step()
                opt_res2.zero_grad()
                res_losses.append(loss.item())
                step_res2 += 1
                if step_res2 >= 6:
                    break
            if step_res2 >= 6:
                break

        # Check losses and final weights
        self.assertEqual(len(cont_losses), len(res_losses))
        for l_c, l_r in zip(cont_losses, res_losses):
            self.assertAlmostEqual(l_c, l_r, places=5)

        for p_name in model_cont.state_dict():
            diff = (model_cont.state_dict()[p_name] - model_res2.state_dict()[p_name]).abs().max().item()
            self.assertLess(diff, 1e-6, f"Weight discrepancy for {p_name}: max diff {diff}")


if __name__ == "__main__":
    unittest.main()
