import numpy as np
from westpa.core.binning import FuncBinMapper


def map_mab(coords, mask, output, *args, **kwargs):
    """
    Construct a set of WE bins according to the MAB scheme.

    Adaptively places bins based on the positions of extrema segments and bottleneck segments, which are where the
    difference in probability is the greatest along the progress coordinate.
    Operates per dimension and places a fixed number of evenly spaced bins between the segments with the min and max
    pcoord values. Extrema and bottleneck segments are assigned their own bins.

    Parameters
    ----------
    coords
    mask
    output
    args
    kwargs

    Returns
    -------
    array-like: The WE bin assignments for each segment.
    """

    # TODO: Output here needs to match the size of the coordinates you're building the bins on
    # TODO: Mask, same deal
    mab_parameters = generate_mab_bins(coords, mask, output, *args, **kwargs)

    # TODO: Output here needs to match the size of the coordinates you're assigning to
    # TODO: Mask, same deal
    assignments = assign_to_mab_bins(coords, mab_parameters, mask, output, *args, **kwargs)

    return assignments


def assign_to_mab_bins(coords, mab_parameters, mask, output, *args, **kwargs):
    """
    Given a set of parameters for the MAB scheme, assign coordinates to bins.

    Parameters
    ----------
    coords
    mab_parameters
    mask
    output

    Returns
    -------
    array-like: The WE bin assignments for each segment.
    """

    #### Things where size matters, and I can't reuse from the generate_bins:
    # - isfinal
    # - output
    # - mask

    allcoords = np.copy(coords)
    allmask = np.copy(mask)

    splitting, ndim, bottleneck, nbins_per_dim, difflist, flipdifflist, minlist, maxlist, isfinal = mab_parameters

    # The base number of bins is the total number of linear bins + 2 boundary bins in each dim
    boundary_base = np.prod(nbins_per_dim)
    bottleneck_base = boundary_base + 2 * ndim

    # Iterate over segments and assign them to bins (if their mask is True)
    for i in range(len(output)):

        # If this segment isn't to be assigned, move on
        if not allmask[i]:
            continue

        special = False
        holder = 0
        if splitting:

            # For this segment, go through each dimension
            for n in range(ndim):
                coord = allcoords[i][n]

                # Handle binning bottleneck segments
                if bottleneck:

                    # If this is the coordinate with the smallest relative log-weight in this dimension, then mark it,
                    #   and store the bin
                    if coord == difflist[n]:
                        # Holder stores the bin index
                        holder = bottleneck_base + 2 * n
                        # This is True for extrema / bottlenecks
                        special = True
                        break
                    elif coord == flipdifflist[n]:
                        holder = bottleneck_base + 2 * n + 1
                        special = True
                        break

                # Handle binning extrema
                if coord == minlist[n]:
                    holder = boundary_base + 2 * n
                    special = True
                    break
                elif coord == maxlist[n]:
                    holder = boundary_base + 2 * n + 1
                    special = True
                    break

        # If this segment is neither a bottleneck or an extremum, then bin it in linear bins
        if not special:

            for n in range(ndim):
                coord = allcoords[i][n]
                nbins = nbins_per_dim[n]
                minp = minlist[n]
                maxp = maxlist[n]

                # Linearly chop up the space between the minimum and maximum coordinates in this set, and assign
                bins = np.linspace(minp, maxp, nbins + 1)
                bin_number = np.digitize(coord, bins) - 1

                # If these are all initial segments and there are no final segments
                if isfinal is None or not isfinal[i]:
                    # Equivalently, bin_number = np.clip(bin_number, 0, nbins-1)
                    if bin_number >= nbins:
                        bin_number = nbins - 1
                    elif bin_number < 0:
                        bin_number = 0

                elif bin_number >= nbins or bin_number < 0:
                    raise ValueError("Walker out of boundary")

                # The bin-index is a scalar, so update it given the number of bins in each dimension
                holder += bin_number * np.prod(nbins_per_dim[:n])

        # Store the bin assignment
        output[i] = holder

    return output


def get_final(coords, allcoords, mask, ndim):
    """
    Determine which segments are "final".
    """

    if coords.shape[1] > ndim:

        # If there's an extra dimension, then we've gotten the segment weights from the MAB driver.
        if coords.shape[1] > ndim + 1:

            # This last index is 1 if it's a final segment, or 0 otherwise, so cast that to a boolean.
            isfinal = allcoords[:, ndim + 1].astype(np.bool_)

        # This else is equivalent to `if coords.shape[1] == ndim + 1`.
        # If that's the case, then we're missing either segment weights or the boolean indicating whether they're final,
        #   which means they're all final coordinates. (Why? When would this happen?)
        else:
            # They're all final coordinates
            isfinal = np.ones(coords.shape[0], dtype=np.bool_)

        # Set coords to hold only the pcoords of the final segments
        coords = coords[isfinal, :ndim]
        # Weights holds the segment weights of the final segments
        weights = allcoords[isfinal, ndim + 0]
        # Only select out the elements of the mask corresponding to final segments
        mask = mask[isfinal]

        splitting = True

        return coords, weights, mask, splitting, isfinal


def generate_mab_bins(coords, mask, output, *args, **kwargs):
    """
    Binning which adaptively places bins based on the positions of extrema segments and
    bottleneck segments, which are where the difference in probability is the greatest
    along the progress coordinate. Operates per dimension and places a fixed number of
    evenly spaced bins between the segments with the min and max pcoord values. Extrema and
    bottleneck segments are assigned their own bins.

    Parameters
    ----------
    coords: array-like
        Set of coordinates to map to WE bins. Shape is (n_segments, n_dimensions + 1), where the extra dimension is
        a boolean indicating whether it's a final segment or not.

    mask: array-like
        A mask indicating which coordinates to assign bins to.

    output: array-like
        Array to populate with the mapped WE bins.

    nbins_per_dim: array-like
        Array storing the number of WE bins to place in each dimension.

    pca: boolean, optional (default: False)
        Whether to run PCA on the components before assignment.

    bottleneck: boolean, optional (default: True)
        Whether to bin bottleneck segments.

    Returns
    -------
    MAB parameters: (splitting, ndim, bottleneck, nbins_per_dim, difflist, flipdifflist, minlist, maxlist, isfinal)
    """

    pca = kwargs.pop("pca", False)
    bottleneck = kwargs.pop("bottleneck", True)
    nbins_per_dim = kwargs.get("nbins_per_dim")
    ndim = len(nbins_per_dim)

    if not np.any(mask):
        return output

    allcoords = np.copy(coords)
    allmask = np.copy(mask)

    weights = None
    isfinal = None
    splitting = False

    # the segments should be sent in by the driver as half initial segments and half final segments
    # allcoords contains all segments
    # coords should contain ONLY final segments
    coords, weights, mask, splitting, isfinal = get_final(coords, allcoords, mask, ndim)

    # in case where there is no final segments but initial ones in range
    if not np.any(mask):
        coords = allcoords[:, :ndim]
        mask = allmask
        weights = None
        splitting = False

    # If PCA is enabled, then do PCA on the coordinates before assignment
    varcoords = np.copy(coords)
    originalcoords = np.copy(coords)
    if pca and len(output) > 1:
        colavg = np.mean(coords, axis=0)
        for i in range(len(coords)):
            for j in range(len(coords[i])):
                varcoords[i][j] = coords[i][j] - colavg[j]
        covcoords = np.cov(np.transpose(varcoords))
        eigval, eigvec = np.linalg.eigh(covcoords)
        eigvec = eigvec[:, np.argmax(np.absolute(eigvec), axis=1)]
        for i in range(len(eigvec)):
            if eigvec[i, i] < 0:
                eigvec[:, i] = -1 * eigvec[:, i]
        for i in range(ndim):
            for j in range(len(output)):
                coords[j][i] = np.dot(varcoords[j], eigvec[:, i])

    maxlist = []
    minlist = []

    # List of the difference of the logs of each segment's weight to the log of all the weight to the right of it,
    #   or (for flipdifflist) to the left of it
    difflist = []
    flipdifflist = []
    for n in range(ndim):
        # identify the boundary segments
        maxcoord = np.max(coords[mask, n])
        mincoord = np.min(coords[mask, n])
        maxlist.append(maxcoord)
        minlist.append(mincoord)

        # detect the bottleneck segments, this uses the weights
        # Splitting is False if there are no final segments provided, only initial segments
        if splitting:
            # Make temp an array of [[segment 0 pcoord, segment 0 weight], ...]
            # Remember that this is just a pcoord in a single dimension, we're looping over dimensions
            temp = np.column_stack((originalcoords[mask, n], weights[mask]))
            sorted_indices = temp[:, 0].argsort()

            temp = temp[sorted_indices]

            # Set 0 weights to a very small number -- necessary because we're taking logs later.
            # This is equivalently temp[ temp[:,1] == 0 ] = 1e-39
            for p in range(len(temp)):
                if temp[p][1] == 0:
                    temp[p][1] = 10 ** -39

            fliptemp = np.flipud(temp)

            # Equivalent to initializing these lists as [None for _ in range(ndim)]
            difflist.append(None)
            flipdifflist.append(None)

            # These aren't actually used
            maxdiff = 0
            flipmaxdiff = 0

            # Loop over segments to find the largest
            for i in range(1, len(temp) - 1):
                comprob = 0
                flipcomprob = 0
                j = i + 1

                # Get the total weight to the left and right of this segment, not including the segment
                # I think this loop is equivalently
                #   comprob = sum(temp[i+1:, 1])
                #   flipcomprob = sum(temp[:i])
                while j < len(temp):
                    comprob = comprob + temp[j][1]
                    flipcomprob = flipcomprob + fliptemp[j][1]
                    j = j + 1

                # Compute log(total weight to the right / segment weight)
                # Store the coordinates of the segment with the highest ratio of this in this dimension,
                #   i.e. the smallest weight relative to the total weight
                # We do this for both "directions" (i.e. with the fliplist as well) so that we can catch bottlenecks
                #   on both the leading and trailing edge.
                diff = -np.log(comprob) + np.log(temp[i][1])
                if diff > maxdiff:
                    difflist[n] = temp[i][0]
                    maxdiff = diff

                # Compute log(total weight to the left / segment weight)
                flipdiff = -np.log(flipcomprob) + np.log(fliptemp[i][1])
                if flipdiff > flipmaxdiff:
                    flipdifflist[n] = fliptemp[i][0]
                    flipmaxdiff = flipdiff

    return splitting, ndim, bottleneck, nbins_per_dim, difflist, flipdifflist, minlist, maxlist, isfinal


class MABBinMapper(FuncBinMapper):
    """
    Adaptively place bins in between minimum and maximum segments along
    the progress coordinte. Extrema and bottleneck segments are assigned
    to their own bins.
    """

    def __init__(self, nbins, bottleneck=True, pca=False):
        kwargs = dict(nbins_per_dim=nbins, bottleneck=bottleneck, pca=pca)
        ndim = len(nbins)
        n_total_bins = np.prod(nbins) + ndim * (2 + 2 * bottleneck)
        super().__init__(map_mab, n_total_bins, kwargs=kwargs)
