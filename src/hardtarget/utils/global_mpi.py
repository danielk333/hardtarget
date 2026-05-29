from typing import Any, Protocol


class CommObject(Protocol):
    rank: int
    size: int

    def bcast(self, obj: Any, root: int = 0) -> Any: ...
    def gather(self, sendobj: Any, root: int = 0) -> list[Any] | None: ...
    def Split(self, color: int, key: int) -> "CommObject": ...
    def barrier(self) -> None: ...


class CommMock(CommObject):
    rank = 0
    size = 1

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def gather(self, sendobj: Any, root: int = 0) -> list[Any]:
        return [sendobj]

    def Split(self, color: int, key: int) -> CommObject:
        return self

    def barrier(self) -> None:
        return None


_COMM: CommObject = CommMock()
_IMPORTED = False


def get_mpi() -> CommObject:
    global _COMM, _IMPORTED
    if not _IMPORTED:
        try:
            from mpi4py import MPI

            _COMM = MPI.COMM_WORLD
        except ImportError:
            _COMM = CommMock()
        _IMPORTED = True
    return _COMM
