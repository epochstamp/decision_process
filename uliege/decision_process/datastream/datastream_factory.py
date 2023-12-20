from .datastream import Datastream  # pragma: no cover
from abc import abstractmethod, ABC  # pragma: no cover


class DatastreamFactory(ABC):  # pragma: no cover
    """
        Interface for objects able to fetch
        distinct datastreams given a ticking index
        for controllers and simulators.
    """

    @abstractmethod
    def _get_controller_datastream(self, tick: int = 0, control_time_horizon: int = 1) -> Datastream:
        """
            Get datastream available at a given tick for the controller
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_simulation_datastream(self, tick: int = 0) -> Datastream:
        """
            Get datastream available at a given tick for the simulation
        """
        raise NotImplementedError()

    def get_controller_datastream(self, tick: int = 0, control_time_horizon: int = 1) -> Datastream:
        """
            Get datastream available at a given tick for the controller

            Parameters
            ----------
            tick: int
                A positive integer

            Returns
            ----------
            Datastream
                A datastream
        """
        if tick < 0:
            raise ValueError("Tick must be positive")
        if control_time_horizon < 1:
            raise ValueError("Control time horizon must be strictly positive")
        return self._get_controller_datastream(tick, control_time_horizon)

    def get_simulation_datastream(self, tick: int = 0):
        """
            Get datastream available at a given tick for the simulation

            Parameters
            ----------
            tick: int
                A positive integer

            Returns
            ----------
            Datastream
                A datastream
        """
        if tick < 0:
            raise ValueError("tick must be positive")
        return self._get_simulation_datastream(tick)
