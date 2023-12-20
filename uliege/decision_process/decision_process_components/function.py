# -*- coding: UTF-8 -*-

from typing import Union
from .evaluator import Evaluator
from abc import ABC, abstractmethod


class Function(ABC):
    """ Interface for functions (formula-based or black-box)
    """

    @abstractmethod
    def __call__(self, evaluator: Evaluator) -> Union[int, float, bool]:
        """Evaluates the function with the object Evaluator

        Parameters
        ----------
        evaluator: :obj:`Evaluator`
            Symbol evaluator

        Returns
        ---------
        :obj:`int`
        | :obj:`float`
        | :obj:`bool`
        Evaluation of the function

        """
        raise NotImplementedError()

    @abstractmethod
    def is_known(self) -> bool:
        """Whether the expression behind the function is known

        Returns
        ---------
        :obj:`bool`
            True if the expression is known
        """
        raise NotImplementedError()
