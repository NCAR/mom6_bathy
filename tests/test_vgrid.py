import pytest
import tempfile
import socket
import numpy as np
import xarray as xr

from mom6_bathy.vgrid import VGrid, _cell_center_to_layer_thickness, _cell_interface_to_layer_thickness

def _create_vgrid_dataset(layer_thickness, cell_center, cell_interface, var_names = ("layer_thickness", "cell_center", "cell_interface")):
    """Create a dataset from the given vgrid information"""
    ds = xr.Dataset({
        var_names[0]: (('z_test',), layer_thickness),
        var_names[1]: (('z_test',), cell_center),
        var_names[2]: (('z_i_test',), cell_interface),})
    
    return ds

def test_default_init(get_realistic_vgrid_elements, get_example_vgrid_elements, get_faulty_vgrid_elements):
    """Test default init behavior for a VGrid object."""
    # Initialize realistic vgrid object
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    vgrid = VGrid(dz = layer_thickness)
    assert np.allclose(vgrid.dz, layer_thickness)
    assert np.allclose(vgrid.z, cell_center)
    assert np.allclose(vgrid.zi, cell_interface)
    
    # Initialize example vgrid object
    layer_thickness, cell_center, cell_interface = get_example_vgrid_elements
    vgrid = VGrid(dz = layer_thickness)
    assert np.allclose(vgrid.dz, layer_thickness)
    assert np.allclose(vgrid.z, cell_center)
    assert np.allclose(vgrid.zi, cell_interface)
    
    # Test faulty data, should raise assertion error
    layer_thickness, cell_center, cell_interface = get_faulty_vgrid_elements
    with pytest.raises(AssertionError):
        vgrid = VGrid(dz = layer_thickness)
    
def test_cell_center_to_layer_thickness(get_realistic_vgrid_elements, get_example_vgrid_elements, get_faulty_vgrid_elements):
    """Test conversion between cell center depths and layer thickness."""
    # Realistic
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    dz = _cell_center_to_layer_thickness(cell_center)
    assert np.allclose(dz,layer_thickness)
    
    # Example
    layer_thickness, cell_center, cell_interface = get_example_vgrid_elements
    dz = _cell_center_to_layer_thickness(cell_center)
    assert np.allclose(dz,layer_thickness)
    
    # Assertion Error
    layer_thickness, cell_center, cell_interface = get_faulty_vgrid_elements
    with pytest.raises(AssertionError):
        dz = _cell_center_to_layer_thickness(cell_center)
    
def test_cell_interface_to_layer_thickness(get_realistic_vgrid_elements, get_example_vgrid_elements, get_faulty_vgrid_elements):
    """Test cell interface depth to layer thickness conversion."""
    # Realistic
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    dz = _cell_interface_to_layer_thickness(cell_interface)
    assert np.allclose(dz,layer_thickness)
    
    # Example
    layer_thickness, cell_center, cell_interface = get_example_vgrid_elements
    dz = _cell_interface_to_layer_thickness(cell_interface)
    assert np.allclose(dz,layer_thickness)
    
    # Assertion Error
    layer_thickness, cell_center, cell_interface = get_faulty_vgrid_elements
    with pytest.raises(AssertionError):
        dz = _cell_interface_to_layer_thickness(cell_interface)
        
def test_from_file_layer_thickness(get_realistic_vgrid_elements, get_example_vgrid_elements):
    """Test basic from_file functionality, loading from layer_thickness with different variable names."""
    
    # Default Variable Names/layer thickness
    layer_thickness, cell_center, cell_interface = get_example_vgrid_elements
    ds = _create_vgrid_dataset(layer_thickness, cell_center, cell_interface, var_names=("dz", "z", "zi"))
        
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()
        
        vgrid = VGrid.from_file(filename=tmpfile.name)
        
        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.z, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)
    
    # Realistic
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    ds = _create_vgrid_dataset(layer_thickness, cell_center, cell_interface)
    
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()
        
        vgrid = VGrid.from_file(
            filename=tmpfile.name, 
            variable_name='layer_thickness',
            variable_type='layer_thickness')
        
        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.z, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)
        
    # Example
    layer_thickness, cell_center, cell_interface = get_example_vgrid_elements
    ds = _create_vgrid_dataset(layer_thickness, cell_center, cell_interface, var_names = ("DZ", "Z", "ZI"))
        
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()
        
        vgrid = VGrid.from_file(
            filename=tmpfile.name, 
            variable_name='DZ',
            variable_type='layer_thickness')
        
        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.z, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)

    
def test_from_file_cell_center(get_realistic_vgrid_elements, get_example_vgrid_elements):
    """Test loading from file with cell center depth information."""
    
    # Realistic
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    ds = _create_vgrid_dataset(layer_thickness, cell_center, cell_interface)
    
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()
        
        vgrid = VGrid.from_file(
            filename=tmpfile.name, 
            variable_name='cell_center',
            variable_type='cell_center')
        
        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.z, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)
        
    # Example
    layer_thickness, cell_center, cell_interface = get_example_vgrid_elements
    ds = _create_vgrid_dataset(layer_thickness, cell_center, cell_interface, var_names=("dz", "z", "zi"))
        
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()
        
        vgrid = VGrid.from_file(
            filename=tmpfile.name, 
            variable_name='z',
            variable_type='cell_center')
        
        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.z, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)
    
def test_from_file_cell_interface(get_realistic_vgrid_elements, get_example_vgrid_elements):
    """Test loading from file with cell interface depth data."""
    
    # Realistic
    layer_thickness, cell_center, cell_interface = get_realistic_vgrid_elements
    ds = _create_vgrid_dataset(layer_thickness, cell_center, cell_interface)
    
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()
        
        vgrid = VGrid.from_file(
            filename=tmpfile.name, 
            variable_name='cell_interface',
            variable_type='cell_interface')
        
        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.z, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)
        
    # Example
    layer_thickness, cell_center, cell_interface = get_example_vgrid_elements
    ds = _create_vgrid_dataset(layer_thickness, cell_center, cell_interface, var_names=("dz", "z", "zi"))
        
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmpfile:
        ds.to_netcdf(tmpfile.name)
        ds.close()
        
        vgrid = VGrid.from_file(
            filename=tmpfile.name, 
            variable_name='zi',
            variable_type='cell_interface')
        
        assert np.allclose(vgrid.dz, layer_thickness)
        assert np.allclose(vgrid.z, cell_center)
        assert np.allclose(vgrid.zi, cell_interface)

if __name__ == "__main__":
    test_default_init()
    test_cell_center_to_layer_thickness()
    test_cell_interface_to_layer_thickness()
    test_from_file_layer_thickness()
    test_from_file_cell_center()
    test_from_file_cell_interface()