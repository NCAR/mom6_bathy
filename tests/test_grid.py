from mom6_bathy.grid import Grid
import pytest 

@pytest.fixture
def get_rect_grid():
    grid = Grid(
        resolution=0.1,
        xstart=278.0,
        lenx=4.0,
        ystart=7.0,
        leny=3.0,
        name="panama1",
    )
    return grid
def test_get_rectangular_segment_info(get_rect_grid):
    grid = get_rect_grid
    res = Grid.get_bounding_boxes_of_rectangular_grid(grid)
    assert "east" in res.keys()
    assert "west" in res.keys()
    assert "north" in res.keys()
    assert "south" in res.keys()
    assert "lat_min" in res["east"].keys()