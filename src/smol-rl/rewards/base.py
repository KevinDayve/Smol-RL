from abc import ABC, abstractmethod

class BaseReward(ABC):
    """
    Interface for all rewards.
    Every scorer takes a completion string and a reference answer string,
    returns a float. Et c'est ca :)
    """
    @abstractmethod
    def score(self, completion: str, reference: str) -> float:
        """
        Args:
            completion: The model's full generated output.
            reference: The ground-truth answer string.
        Returns:
            A scalar reward.
        """
        ...