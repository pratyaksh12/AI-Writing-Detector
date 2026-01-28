import random;
from typing import TypeVar, Generic, Iterable;


T = TypeVar("T")

class RandomList(Generic[T]):
    def __init__(self, data: Iterable[T]) -> None:
        self.data: list[T] = list(data)
        
    def append(self, item: T) -> None:
        self.data.append(item)
        
    def pop(self) -> T:
        if not self.data:
            raise IndexError("pop from an empty list")
        
        idx: int=  random.randint(0, len(self.data) - 1)
        self.data[idx], self.data[-1] = self.data[-1], self.data[idx]
        return self.data.pop()
    
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __bool__(self) -> bool:
        return bool(self.data)