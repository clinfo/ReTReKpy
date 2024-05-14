""" The ``retrekpy.in_scope_filter`` package ``meter`` module. """

import numpy


class Meter:
    """ The 'Meter' class. """

    def __init__(
            self,
            coeff: float = 0.95
    ) -> None:
        """ The constructor method of the class. """

        self.coeff = coeff
        self.value = numpy.nan

    def update(
            self,
            value: float
    ) -> None:
        """ The 'update' method of the class. """

        if not self.value == self.value:
            self.value = value

        else:
            self.value = self.coeff * self.value + (1 - self.coeff) * value
