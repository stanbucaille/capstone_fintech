from tensortrade.rewards import RewardScheme


class SimpleProfit(RewardScheme):
    """A simple reward scheme that rewards the agent for incremental increases in net worth."""

    def __init__(self, window_size: int = 1):
        self.window_size = window_size

    def reset(self):
        pass

    def get_reward(self, portfolio: 'Portfolio') -> float:
        """Rewards the agent for incremental increases in net worth over a sliding window.

        Args:
            portfolio: The portfolio being used by the environment.

        Returns:
            The incremental increase in net worth over the previous `window_size` timesteps.
        """
        returns = portfolio.performance['net_worth'].diff()
        return sum(returns[-self.window_size:])
