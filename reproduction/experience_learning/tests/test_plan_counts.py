from experience_learning.pilot import count_nodes,count_leaves


def test_prefix_sharing_counts_both_orders_and_all_tools():
    tools={'a':list(range(5)),'b':list(range(4))}
    assert count_nodes(['a','b'],tools)==49
    assert count_leaves(['a','b'],tools)==40
    assert count_nodes([],tools)==0


def test_three_degradations_are_not_mistaken_for_two():
    tools={'rain':list(range(4)),'noise':list(range(5)),'sr':list(range(5))}
    assert count_leaves(list(tools),tools)==600
    assert count_nodes(list(tools),tools)==744


def test_ridcp_excluded_pilot_budget():
    tools={"noise":list(range(5)),"blur":list(range(4)),"rain":list(range(4)),"haze":list(range(4))}
    assert 2*(count_nodes(["noise","blur"],tools)+count_nodes(["rain","haze"],tools))==178
    assert 2*(count_leaves(["noise","blur"],tools)+count_leaves(["rain","haze"],tools))==144
