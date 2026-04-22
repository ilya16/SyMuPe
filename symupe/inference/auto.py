import torch

from symupe.data.tokenizers import MusicTokenizer
from symupe.models import Model
from .base import Generator, Classifier


class _AutoFactory:
    """Internal factory class for initializing inference wrappers."""

    @classmethod
    def from_model(
        cls,
        model: Model,
        tokenizer: MusicTokenizer,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> Generator | Classifier:
        """Initializes inference wrapper from existing model and tokenizer instances.

        Args:
            model: :class:`Model` instance to wrap.
            tokenizer: :class:`MusicTokenizer` instance for data processing.
            device: Target device for inference.
            **kwargs: Additional parameters for inference wrapper.

        Returns:
            Initialized Generator or Classifier instance.
        """
        ...

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> Generator | Classifier:
        """Initializes inference wrapper by loading model and tokenizer from path.

        Args:
            pretrained_path: Path or Hub repo ID for pretrained artifacts.
            device: Target device for inference.
            **kwargs: Additional parameters for model or inference wrapper.

        Returns:
            Initialized Generator or Classifier instance.
        """
        from symupe.models import AutoModel
        from symupe.data.tokenizers import AutoTokenizer

        # load Model
        model = AutoModel.from_pretrained(pretrained_path, **kwargs)
        if device is not None:
            model.to(device)

        # load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

        # initialize Generator
        return cls.from_model(model=model, tokenizer=tokenizer, device=device, **kwargs)


class AutoGenerator(_AutoFactory):
    """Factory class to initialize sequence generators."""

    @classmethod
    def from_model(
        cls,
        model: Model,
        tokenizer: MusicTokenizer,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> Generator:
        """Creates task-specific generator based on model architecture.

        Args:
            model: :class:`Model` instance to wrap.
            tokenizer: :class:`MusicTokenizer` instance for data processing.
            device: Target device for inference.
            **kwargs: Additional parameters for generator.

        Returns:
            Initialized Generator instance.
        """
        # get Generator class from the model
        if model.GENERATOR_CLASS is None:
            model_name = model.__class__.__name__
            raise ValueError(f"Model of class {model_name} does not have a Generator class")

        generator_cls = model.GENERATOR_CLASS

        return generator_cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )


class AutoClassifier(_AutoFactory):
    """Factory class to initialize classifiers."""

    @classmethod
    def from_model(
        cls,
        model: Model,
        tokenizer: MusicTokenizer,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> Classifier:
        """Creates task-specific classifier based on model architecture.

        Args:
            model: :class:`Model` instance to wrap.
            tokenizer: :class:`MusicTokenizer` instance for data processing.
            device: Target device for inference.
            **kwargs: Additional parameters for classifier.

        Returns:
            Initialized Classifier instance.
        """
        # get Classifier class from the model
        if model.CLASSIFIER_CLASS is None:
            model_name = model.__class__.__name__
            raise ValueError(f"Model of class {model_name} does not have a Classifier class")

        classifier_cls = model.CLASSIFIER_CLASS

        return classifier_cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )
