from dataclasses import dataclass

@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (
            self.left,
            self.top,
            self.left + self.width,
            self.top + self.height,
        )
