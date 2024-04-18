#!C:\Python27\python.exe
try:
    import os,sys,argparse
    sys.path.append(r'C:\Program Files (x86)\Gwyddion\bin')  # Windows install
    sys.path.append(r'C:\Program Files (x86)\Gwyddion\share\gwyddion\pygwy')  # pygwy location on Windows

    import pygtk
    pygtk.require20()  # adds gtk-2.0 folder to sys.path
    
    import gwy
    import numpy as np
    import gwyutils
    from glob import glob
    from pathlib2 import Path
    import tifffile
    import traceback

    parser = argparse.ArgumentParser(description="Surface processing with pygwy.")
    parser.add_argument('files', nargs='+', type=str)
    parser.add_argument('-o', '--target', type=str)
    parser.add_argument('-t','--type', type=str, default='tiff')
    parser.add_argument('-i', '--invalid', type=float, default=None)
    parser.add_argument('-m', '--mask', type=str, default=None)

    args = vars(parser.parse_args())

    files = args['files']
    target_path = Path(args['target'])
    target_path.mkdir(exist_ok=True, parents=True)

    filetype = args['type']

    if args['invalid'] != None:
        remove_invalids = True
        invalid = args['invalid']
    else:
        remove_invalids = False

    mask_type = args['mask']
    if mask_type and not mask_type in ['outliers']:
        print("Error: Mask type unknown! Not producing mask.")
        mask_type = None


    print("Evaluating Surface Data ...")
    print("From files:\n" + str(files))
    print("Target path: " + str(target_path))

    settings = gwy.gwy_app_settings_get()

    for f in files:

        try:
            print("Loading file " + str(f))
            new_filename = target_path / Path(f).stem
            #print("Target Path: " + new_filename)

            container = gwy.gwy_file_load(f, gwy.RUN_IMMEDIATE)
            gwy.gwy_app_data_browser_add(container)
            data_field_idx = gwy.gwy_app_data_browser_find_data_by_title(container, "Height")[0]
            print(data_field_idx)
            gwy.gwy_app_data_browser_select_data_field(container, data_field_idx)
            
            
            print("Applying transformations")

            gwy.gwy_process_func_run('flatten_base', container, gwy.RUN_IMMEDIATE)
        
            
            data_field = container['/{0}/data'.format(data_field_idx)]


            size = (data_field.get_xreal(), data_field.get_yreal())
            res = (data_field.get_xres(), data_field.get_yres())
            
            dx = data_field.get_dx()
            dy = data_field.get_dy()
            
            val_lim = data_field.get_min_max()
            
            print("Smoothing")
            # filter to remove noise
            data_field.filter_gaussian(1.0)

            print("Exporting results ...")
            
            gwy.gwy_app_data_browser_select_data_field(container, data_field_idx)

            dxu, dyu = dx*1e6, dy*1e6
            gwy_meta = container['/{0}/meta'.format(data_field_idx)]
            # print(str(gwy_meta.serialize_to_text()))
            meta_dict = {k:gwy_meta.get_value_by_name(k) for k in gwy_meta.keys_by_name()}

            np_data = gwyutils.data_field_data_as_array(data_field)
            metadata=dict(microscope="Olympus LEXT 4000", PhysicalSizeX=dxu, PhysicalSizeY=dyu, PhysicalSizeUnit="um", PhysicalSizeZ=0.8, **meta_dict)
            # print(new_filename,new_filename.with_suffix(".ome.tif"))
            print(1/dx, 1/dy)
            tifffile.imwrite(str(new_filename.with_suffix(".ome.tif")), np_data, 
                    compress=('zstd',9), tile=(256,256),
                    metadata=metadata, ijmetadata=metadata,
                    resolution=(1/dxu, 1/dyu))
            
            gwy.gwy_app_data_browser_remove(container)
            print("Done")
            
        except Exception as ex:
            print(traceback.format_exc())
            
except Exception as ex:
    print(str(ex))