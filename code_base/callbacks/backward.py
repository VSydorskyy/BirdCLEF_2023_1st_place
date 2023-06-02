from catalyst import callbacks


class BackwardCallbackFixed(callbacks.BackwardCallback):
    def on_batch_end(self, runner: "IRunner"):
        """Event handler."""
        if runner.is_train_loader:
            loss = runner.batch_metrics[self.metric_key]
            runner.engine.backward(loss)
            if self.grad_clip_fn is not None:
                runner.engine.unscale_gradients()
                norm = self.grad_clip_fn(runner.model.parameters())
                if self._log_gradient:
                    runner.batch_metrics[
                        f"{self._prefix_gradient}/norm"
                    ] = norm
