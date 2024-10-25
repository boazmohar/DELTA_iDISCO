import re

def assign_layer_numbers(atlas):
    """
    Assigns layer numbers to brain regions based on the region name.
    
    Parameters:
    - atlas (BrainGlobeAtlas): BrainGlobeAtlas object containing the brain region hierarchy.

    Returns:
    - dict: Dictionary mapping each brain region acronym to its layer number. 
    
    Notes:
    - Layer 3 is combined with layer 2. 
    - Layer 0 is assigned if no layer information is found in the region name. 
    - Layer numbers are integers so the a and b of layer 5 and 6 are combined.
    """
    layers = {}
    layer_pattern = re.compile(r'[lL]ayer\s(\d)')

    for region in atlas.structures.values():
        match = layer_pattern.search(region['name'])
        if match:
            layer = int(match.group(1))
            # Combine layer 3 into layer 2 as per the MATLAB code
            if layer == 3:
                layer = 2
            layers[region['acronym']] = layer
        else:
            layers[region['acronym']] = 0  # No layer information found

    return layers


def map_to_large_regions(atlas, small_region_list=None, large_region_names=None):
    """
    Maps small brain regions to larger brain regions based on atlas hierarchy.
    
    Parameters:
    - small_region_list (list): List of small brain region acronyms or names to map. defaults to all regions in the atlas. 
      needs to be a list of acronyms of the regions in the atlas.
    - large_region_names (list): List of brain region names for classification. defaults to 12 major regions.
    - atlas (BrainGlobeAtlas): BrainGlobeAtlas object containing the brain region hierarchy.
    
    Returns:
    - dict: Dictionary mapping each small region to one of the larger regions. Would return 'root' if no large region is found.
    """
    if small_region_list is None:
        small_region_list = [region['acronym'] for region in atlas.structures.values()]  # Get all regions in the atlas
    if large_region_names is None:
        large_region_names = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", 
                              "TH", "HY", "MB", "P", "CB", "MY", "VS", "fiber tracks"]
    # Create the mapping
    region_mapping = {}
    for small_region in small_region_list:
        # Ensure region exists in the atlas and is not itself a large region
        if small_region in large_region_names:
            region_mapping[small_region] = small_region
        else:
            try:
                # Get ancestors of the region by name using BrainGlobeAtlas function
                ancestors = atlas.get_structure_ancestors(small_region)
                
                # Find the first matching ancestor in large_region_names (case-insensitive partial match)
                found_large_region = next((ancestor for ancestor in ancestors if any(large_region.lower() in ancestor.lower() for large_region in large_region_names)), "root")
                region_mapping[small_region] = found_large_region
            except KeyError:
                region_mapping[small_region] = "root"
    
    return region_mapping

def acronyms_to_names(atlas, acronym_list):
    names = [region['name'] for region in atlas.structures.values() if region['acronym'] in acronym_list]
    return names