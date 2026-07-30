import pathlib
import tensorflow.keras as keras
from datetime import datetime
import string


class Callback:
    """Abstract base class (ABC) for training callbacks.
    
    This ABC is loosely based on the `keras.callbacks.Callback` API, but overloaded to allow training in reinforcement learning settings and for frameworks other than TensorFlow.
    """
    
    def __init__(self):
        self._algorithm = None

    @property
    def algorithm(self):
        return self._algorithm
    
    @algorithm.setter
    def algorithm(self, algorithm):
        self._algorithm = algorithm

    def on_episode_begin(self, episode: int):
        """Called at the start of a reinforcement learning episode."""
        pass
    
    def on_episode_end(self, episode: int):
        """Called at the end of a reinforcement learning episode."""
        pass
    
    def on_train_begin(self):
        """Called at the start of reinforcement learning training."""
        pass
    
    def on_train_end(self):
        """Called at the end of reinforcement learning training."""
        pass


class CallbackList(Callback):
    """Wrapper class for working with a list of callbacks.
    
    Allows invoking functions from a list of callback objects using a single call to the entire list.
    """
    
    def __init__(self, callbacks: list[Callback] = None):
        super().__init__()
        self.callbacks = callbacks or []
        
    def __getitem__(self, item):
        return self.callbacks[item]
        
    def _recursive_call_children(self, func: str, *args, **kwargs):
        """Call methods of all children callback class instances."""
        for cb in self.callbacks:
            f = getattr(cb, func)
            f(*args, **kwargs)
    
    def append(self, callback):
        self.callbacks.append(callback)
        
    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        self._algorithm = algorithm
        for cb in self.callbacks:
            cb.algorithm = algorithm

    def on_episode_begin(self, episode: int):
        """Called at the start of a reinforcement learning episode."""
        self._recursive_call_children(self.on_episode_begin.__name__, episode=episode)
    
    def on_episode_end(self, episode: int):
        """Called at the end of a reinforcement learning episode."""
        self._recursive_call_children(self.on_episode_end.__name__, episode=episode)

    def on_train_begin(self):
        """Called at the start of reinforcement learning training."""
        self._recursive_call_children(self.on_train_begin.__name__)
    
    def on_train_end(self):
        """Called at the end of reinforcement learning training."""
        self._recursive_call_children(self.on_train_end.__name__)




class TensorflowModelCheckpoint(Callback):
    """Save Tensorflow model checkpoint periodically during training.
    
    Save interval is tracked at the end of each episode.
    """
    
    def __init__(self, model: keras.Model, filepath: pathlib.Path, save_freq: int, save_weights_only: bool = True, verbose: bool = False):
        super().__init__()
        self.model = model
        self.filepath = filepath
        self.save_freq = save_freq
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        assert isinstance(save_freq, int), 'save frequency must be an integer number of episodes'
        
    def on_episode_end(self, episode: int):
        if ((episode+1) % self.save_freq) == 0:
            fps = str(self.filepath).format(
                datetime=datetime.now().isoformat(),
                episode=episode+1,
                )
            self.model.save_weights(fps)
            if self.verbose:
                print(f"Saving model {self.model.name} at episode {episode+1} to file {fps}")



class AlgorithmModelCheckpoint(Callback):
    """Save algorithm model checkpoint periodically during training.
    
    Save interval is tracked at the end of each episode.
    """
    
    def __init__(self, model_name: str, filepath: pathlib.Path, save_freq: int, save_weights_only: bool = True, verbose: bool = False):
        super().__init__()
        self.model_name = model_name
        self.filepath = filepath
        self.save_freq = save_freq
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        assert isinstance(save_freq, int), 'save frequency must be an integer number of episodes'
        
    def on_episode_end(self, episode: int):
        if ((episode+1) % self.save_freq) == 0:
            fmt_keys = [tup[1] for tup in string.Formatter().parse(str(self.filepath)) if tup[1] is not None]
            
            fmt_dict = {}
            if 'episode' in fmt_keys:
                fmt_dict['episode'] = episode+1
                fmt_keys.remove('episode')
            for key in fmt_keys:
                fmt_dict[key] = getattr(self.algorithm, key)

            fps = str(self.filepath).format(**fmt_dict)
            
            self.algorithm.save_model(self.model_name, fps, self.save_weights_only)

            if self.verbose:
                print(f"Saving model {self.model_name} at episode {episode+1} to file {fps}")




class AlgorithmResultCheckpoint(Callback):
    """Save algorithm results periodically during training.
    
    Save interval is tracked at the end of each episode.
    """
    
    def __init__(self, filepath: pathlib.Path, save_freq: int, verbose: bool = False):
        super().__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.verbose = verbose
        
        assert isinstance(save_freq, int), 'save frequency must be an integer number of episodes'

    def on_episode_end(self, episode: int):
        if ((episode+1) % self.save_freq) == 0:
            fmt_keys = [tup[1] for tup in string.Formatter().parse(str(self.filepath)) if tup[1] is not None]
            
            fmt_dict = {}
            if 'episode' in fmt_keys:
                fmt_dict['episode'] = episode+1
                fmt_keys.remove('episode')
            for key in fmt_keys:
                fmt_dict[key] = getattr(self.algorithm, key)

            fps = str(self.filepath).format(
                **fmt_dict,
                # datetime=datetime.now().isoformat(),
                # episode=episode+1,
                )
            self.algorithm.save(filepath=fps)
            if self.verbose:
                print(f"Saving results at episode {episode+1} to file {fps}")


class ActiveSLAMPilotMonitor(Callback):
    """Stop pilot training after sustained policy collapse or coverage loss."""

    def __init__(
        self,
        window: int = 100,
        turn_threshold: float = 0.85,
        max_forward_fraction: float = 0.05,
        coverage_drop: float = 0.10,
        verbose: bool = True,
    ):
        super().__init__()
        self.window = int(window)
        self.turn_threshold = float(turn_threshold)
        self.max_forward_fraction = float(max_forward_fraction)
        self.coverage_drop = float(coverage_drop)
        self.verbose = bool(verbose)

    @staticmethod
    def _mean(records, key):
        return sum(float(record[key]) for record in records) / len(records)

    def on_episode_end(self, episode: int):
        history = self.algorithm.episode_metrics_history
        if len(history) < self.window:
            return
        recent = history[-self.window:]
        left = self._mean(recent, "left_action_fraction")
        right = self._mean(recent, "right_action_fraction")
        forward = self._mean(recent, "forward_action_fraction")
        reason = None
        if max(left, right) >= self.turn_threshold or forward <= self.max_forward_fraction:
            reason = (
                f"sustained turn collapse over {self.window} episodes "
                f"(left={left:.3f}, right={right:.3f}, forward={forward:.3f})"
            )
        elif len(history) >= 2 * self.window:
            previous = history[-2 * self.window:-self.window]
            coverage_change = self._mean(recent, "coverage") - self._mean(previous, "coverage")
            if coverage_change <= -self.coverage_drop:
                reason = (
                    f"coverage fell by {abs(coverage_change):.3f} over consecutive "
                    f"{self.window}-episode windows"
                )
        if reason:
            self.algorithm.stop_training = True
            self.algorithm.stop_reason = reason
            if self.verbose:
                print(f"Pilot monitor: {reason}")
