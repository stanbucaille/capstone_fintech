from abc import abstractmethod

from tensortrade import Component, TimeIndexed


class RewardScheme(Component, TimeIndexed):

    registered_name = "rewards"

    def reset(self):
        """Optionally implementable method for resetting stateful schemes."""
        pass

    @abstractmethod
    def get_reward(self, portfolio: 'Portfolio') -> float:
        """
        Arguments:
            portfolio: The portfolio being used by the environment.

        Returns:
            A float corresponding to the benefit earned by the action taken this timestep.
        """
        raise NotImplementedError()
