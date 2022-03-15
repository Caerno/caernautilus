import operator
from typing import Callable
from numbers import Number

def dull(_a=None,_b=None):return True
oper = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "": dull
}

def number(msg:str="Enter number: ", t:Callable=int, op:str=">=", limit:int=0) -> Number:
    if t == str:
        limit = ""
    func = oper[op]
    x = t(input(msg))
    while not func(x, limit):
        print(f"Condition {x} {op} {limit} not met, try again")
        x = t(input(msg))
    return x