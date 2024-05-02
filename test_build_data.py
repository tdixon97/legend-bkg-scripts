import awkward as ak

import build_data


def test_filter_off_ac():

    ### filter an off detector
    ### possibilites
    ### 1) the off detector is part of the multiplet - hit removed
    ### 2) its not - QC changed
    ### 3) neither
    data = {
        "geds": {
            "cut": [[True, True], [False], [False, False]],
            "is_good_channel": [[True, True], [True], [True, True]],
            "hit_rawid": [[1, 2], [1], [2, 3]],
            "energy": [[100, 200], [500], [100, 100]],
            "unphysical_hit_rawid": [[], [4], [3]],
        }
    }
    data_ak = ak.Array(data)
    ### this should remove the hit in ch3 (third event) and change the qc in the second event
    data_filtered = build_data.filter_off_ac(data_ak, "cut", [], [3, 4])

    assert ak.all(
        data_filtered["geds", "cut"] == ak.Array([[True, True], [True], [True]])
    )
    assert ak.all(
        data_filtered["geds", "is_good_channel"]
        == ak.Array([[True, True], [True], [True]])
    )
    assert ak.all(data_filtered["geds", "hit_rawid"] == ak.Array([[1, 2], [1], [2]]))
    assert ak.all(
        data_filtered["geds", "energy"] == ak.Array([[100, 200], [500], [100]])
    )
    assert ak.all(
        data_filtered["geds", "unphysical_hit_rawid"] == ak.Array([[], [], []])
    )

    ### filter an AC this should just change the cut

    data = {
        "geds": {
            "cut": [[True, True], [True], [True, False]],
            "is_good_channel": [[True, True], [True], [True, False]],
            "hit_rawid": [[1, 2], [4], [2, 3]],
            "energy": [[100, 200], [500], [100, 100]],
            "unphysical_hit_rawid": [[], [], []],
        }
    }
    data_ak = ak.Array(data)

    data_filtered = build_data.filter_off_ac(data_ak, "cut", [4], [])
    print(data_filtered["geds", "is_good_channel"].to_list())
    assert ak.all(
        data_filtered["geds", "cut"] == ak.Array([[True, True], [False], [True, False]])
    )
    assert ak.all(
        data_filtered["geds", "is_good_channel"]
        == ak.Array([[True, True], [False], [True, False]])
    )
