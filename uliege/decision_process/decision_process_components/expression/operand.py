from __future__ import annotations


class Operand:
    """ Abstract class which surcharge usual mathematical operators
        and item getters '[]'
    """

    def _transform_to_numeric_expression(self) -> NumericExpression:
        """ Turn the operand into a numeric expression expression

            Returns
            ----------
            Expression
                itself if already an Expression, IndexedContainer otherwise
                (which inherits from Expression as well)

            Raises
            ----------
            TypeError
                Not possible to transform the operand into a numeric
                expression
                (e.g., one of the operands is a logical expression)
        """
        from ..variable import Variable
        from ..parameter import Parameter
        from .numeric_expression\
            import NumericExpression
        from .indexed_container import\
            IndexedContainer
        if isinstance(self, Variable) or isinstance(self, Parameter):
            idx_container = IndexedContainer(container=self,
                                             idx_seq=tuple())
            return idx_container
        elif isinstance(self, IndexedContainer):
            true_self = self._transform_to_indexed_container()
            return true_self
        elif isinstance(self, NumericExpression):
            return self

    def _transform_to_indexed_container(self: IndexedContainer)\
            -> IndexedContainer:
        from .indexed_container import\
            IndexedContainer
        return self

    def __add__(self, other: Operand) -> Binop:
        """ Build an addition between the
            current expression and the other expression

            Parameters
            ----------
            other : Operand
                The second-hand operand of the operation

            Returns
            ----------
            Binop
                The binary operation between self and other.
        """
        expr_1 = self._transform_to_numeric_expression()
        expr_2 = other._transform_to_numeric_expression()
        from .binop import Binop
        from .binop import Binoperator
        return Binop(expr_1=expr_1, binop=Binoperator.ADD, expr_2=expr_2)

    def __sub__(self, other: Operand) -> Binop:
        """ Build a substraction between the
            current expression and the other expression

            Parameters
            ----------
            other : Operand
                The second-hand operand of the operation

            Returns
            ----------
            Binop
                The binary operation between self and other.
        """
        expr_1 = self._transform_to_numeric_expression()
        expr_2 = other._transform_to_numeric_expression()
        from .binop import Binop
        from .binop import Binoperator
        return Binop(expr_1=expr_1, binop=Binoperator.SUB, expr_2=expr_2)

    def __mul__(self, other: Operand) -> Binop:
        """ Build a product between the
            current expression and the other expression

            Parameters
            ----------
            other : Operand
                The second-hand operand of the operation

            Returns
            ----------
            Binop
                The binary operation between self and other.
        """
        expr_1 = self._transform_to_numeric_expression()
        expr_2 = other._transform_to_numeric_expression()
        from .binop import Binop
        from .binop import Binoperator
        return Binop(expr_1=expr_1, binop=Binoperator.MUL, expr_2=expr_2)

    def __neg__(self) -> Unop:
        """ Returns the negative-signed expression

            Parameters
            ----------
            other : Operand
                The second-hand operand of the operation

            Returns
            ----------
            Unop
                The negation of this expression.
        """
        expr = self._transform_to_numeric_expression()
        from .unop import Unop
        from .unop import Unoperator
        return Unop(expr=expr, unop=Unoperator.NEG)

    def __truediv__(self, other: Operand) -> Binop:
        """ Build a division between the
            current expression and the other expression

            Parameters
            ----------
            other : Operand
                The second-hand operand of the operation

            Returns
            ----------
            Binop
                The binary operation between self and other.
        """
        expr_1 = self._transform_to_numeric_expression()
        expr_2 = other._transform_to_numeric_expression()
        from .binop import Binop
        from .binop import Binoperator
        return Binop(expr_1=expr_1, binop=Binoperator.DIV, expr_2=expr_2)

    def __le__(self, other: Operand) -> Ineq:
        """ Build a lower-or-equal inequation between the expressions

        Parameters
            ----------
            other : Operand
                The second-hand operand of the operation

            Returns
            ----------
            Constraint
                A lower-or-equal constraint between self and other.
        """
        expr_1 = self._transform_to_numeric_expression()
        expr_2 = other._transform_to_numeric_expression()
        from .ineq import\
            Ineq, Ineqoperator
        return Ineq(expr_1=expr_1,
                    ineq_op=Ineqoperator.LE,
                    expr_2=expr_2)

    def __ge__(self, other: Operand) -> Ineq:
        """ Build a greater-or-equal inequation between the expression

        Parameters
            ----------
            other : Operand
                The second-hand operand of the operation

            Returns
            ----------
            Constraint
                A greater-or-equal constraint
                between self and other.
        """
        expr_1 = self._transform_to_numeric_expression()
        expr_2 = other._transform_to_numeric_expression()
        from .ineq import\
            Ineq, Ineqoperator
        return Ineq(expr_1=expr_1,
                    ineq_op=Ineqoperator.GE,
                    expr_2=expr_2)

    def __eq__(self, other: Operand) -> Ineq:
        """ Build an equation between the expressions

        Parameters
            ----------
            other : Operand
                The second-hand operand of the operation

            Returns
            ----------
            Constraint
                An equality constraint between self and other.
        """
        expr_1 = self._transform_to_numeric_expression()
        expr_2 = other._transform_to_numeric_expression()
        from .ineq import\
            Ineq, Ineqoperator
        return Ineq(expr_1=expr_1,
                    ineq_op=Ineqoperator.EQ,
                    expr_2=expr_2)

    def __abs__(self) -> Unop:
        """ Returns the negative-signed expression

            Parameters
            ----------
            other : Operand
                The second-hand operand of the operation

            Returns
            ----------
            Unop
                The negation of this expression.
        """
        expr = self._transform_to_numeric_expression()
        from .unop import Unop
        from .unop import Unoperator
        return Unop(expr=expr, unop=Unoperator.ABS)
