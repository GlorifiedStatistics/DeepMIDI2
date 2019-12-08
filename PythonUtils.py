"""
For future reference so I don't have to keep writing it taking up space:
- Operating on circular lists/dictionaries currently has undefined behavior.
- Some common parameters:
    "comparators": can be set to a list of lists, where each sub-list contains at least three elements:
        - first, the type of the first object being compared
        - second, the type of the second object being compared
        - third, a function, taking two args and *args, that returns a boolean based on whether or not
            the first argument is equal to the second argument
        - Any extra elements are passed to the function as args
        Default: None
    "ordered": if True, the lists and all sublists need to be in the same order.
        Default: False
    "typeless_lists": if True, then lists need not be of the same type, only some type of list.
        Default: False
    "remove_used": if True, then elements in l2 are removed from inspection as they are taken up by elements in l1
        For example - if remove_used is False, then [a, a, a, a] would be equal to [a,], but if it were True,
            this would not be the case
        Default: True
"""


def list_to_string(l):
    """
    Converts the given list of strings into a single string by appending them all together. Calls the str() built-in
    function on every element inside l, so a string is always created.
    """
    if not is_list(l):
        raise TypeError("l must be some type of list, instead is %s" % type(l))
    ret = ""
    for v in l:
        ret += str(v)
    return ret


def dict_to_list(d):
    """
    Converts the given dictionary to a list (removes key info). All sub-dicts will become sub-lists.
    :param d: the dictionary to convert into a list
    :return: the given dictionary as a list
    """
    pass


def linearize_list(l):
    """
    Converts l to be a list containing no sub-lists.
    If l is not a list, l is converted into a list of size 1.
    All empty lists are ignored.
    Dictionaries are not considered lists.
    """
    if not is_list(l):
        return [l, ]

    ret = []
    for element in l:
        if is_list(element):
            ret += linearize_list(element)
        else:
            ret.append(element)

    return ret


def is_list(l):
    """
    Checks iterability and indexing to make see if l is a list
    """
    try:
        for _ in l:
            pass
        if len(l) > 0:
            a = l[0]
        return not isinstance(l, dict)
    except TypeError:
        return False


def is_sublist(l1, l2, proper=False, ordered=False, typeless_lists=False, remove_used=True, comparators=None):
    """
    Returns True if l1 is a sublist of l2. If proper is True, then l1 must be a proper sublist of l2
    """
    if not is_list(l1) or not is_list(l2):
        raise TypeError("Either l1 or l2 are not lists %s, %s" % (type(l1), type(l2)))

    # Check comparators is good
    _check_comparators(comparators)

    # Check these elements are both the same type if we need to
    if not typeless_lists and not isinstance(l1, type(l2)):
        return False

    checked = []
    for e in l1:
        if not _find_object(e, l2, checked if remove_used else [], ordered, typeless_lists, remove_used, comparators):
            return False

    return not cmp_list(l1, l2) if proper else True


def is_subdict(d1, d2, proper=False, ordered=False, typeless_lists=False, remove_used=True, comparators=None):
    """
    Returns True if d1 is a subdict of d2. If proper is True, then d1 must be a proper subdict of d2.
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        raise TypeError("Either d1 or d2 is not a dicitonary %s, %s" % (type(d1), type(d2)))

    _check_comparators(comparators)

    for key in d1.keys():
        if key not in d2.keys():
            return False
        if not _cmp_objects(d1[key], d2[key], ordered, typeless_lists, remove_used, comparators):
            return False

    return len(d1.keys()) != len(d2.keys()) if proper else True


def cmp_list(l1, l2, ordered=False, typeless_lists=False, remove_used=True, comparators=None):
    """
    Returns true if l1 and l2 contain the same elements.
    Dictionaries by default are compared using the cmp_dicts function, with the same parameters as this function.
    """

    if not is_list(l1) or not is_list(l2):
        raise TypeError("Either l1 or l2 are not lists %s, %s" % (type(l1), type(l2)))
    _check_comparators(comparators)

    # Check these elements are both the same type if we need to
    if not typeless_lists and not isinstance(l1, type(l2)):
        return False

    # Check size first
    if remove_used and len(l1) != len(l2):
        return False

    # Do it this way if ordered
    if ordered:
        for i in range(len(l1)):
            if not _cmp_objects(l1[i], l2[i], ordered, typeless_lists, remove_used, comparators):
                return False

    # Otherwise the harder way
    else:
        checked = []  # The list of checked indices in l2
        for i in range(len(l1)):
            # Try to find this object in l2 in an index that has not already been checked
            if not _find_object(l1[i], l2, checked if remove_used else [], ordered, typeless_lists, remove_used,
                                comparators):
                return False

    return True


def cmp_dict(d1, d2, ordered=False, typeless_lists=False, remove_used=True, comparators=None):
    """
    Returns True if the dictionaries d1 and d2 are equivalent.
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        raise TypeError("Either d1 or d2 is not a dicitonary %s, %s" % (type(d1), type(d2)))

    _check_comparators(comparators)

    if len(d1.keys()) != len(d2.keys()):
        return False

    for key in d1.keys():
        if key not in d2.keys():
            return False
        if not _cmp_objects(d1[key], d2[key], ordered, typeless_lists, remove_used, comparators):
            return False
    return True


def diff_list(l1, l2, ordered=False, typeless_lists=False, remove_used=True, comparators=None):
    """
    Returns all of the elements in l1 that are not in l2 (ie: l1 - l2).
    """
    if not is_list(l1) or not is_list(l2):
        raise TypeError("Either l1 or l2 are not lists %s, %s" % (type(l1), type(l2)))
    _check_comparators(comparators)

    checked = []
    return [e for e in l1 if
            not _find_object(e, l2, checked if remove_used else [], ordered, typeless_lists, remove_used, comparators)]


def diff_dict(d1, d2, ordered=False, typeless_lists=False, remove_used=True, comparators=None):
    """
    Returns all of the elements in d1 that are not in d2 (ie: d1 - d2)
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        raise TypeError("Either d1 or d2 is not a dicitonary %s, %s" % (type(d1), type(d2)))

    _check_comparators(comparators)

    ret = {}
    for key in d1.keys():
        if key not in d2.keys() or not _cmp_objects(d1[key], d2[key], ordered, typeless_lists, remove_used,
                                                    comparators):
            ret[key] = d1[key]

    return ret


def add_dicts(d1, d2, behavior=None):
    """
    Combines the two dictionaries together. If they have any elements that share the same key, the behavior is
        defined by 'behavior' which can take the values:
        - 'first' : keep the value from d1, ignore d2
        - 'second' : keep the value from d2, ignore d1
        - 'add' : attempt to add the two values together using the default __add__ implementation
        - None: if behavior is left as None, an error is thrown upon conflicting keys
    :param behavior: how to deal with elements that share the same key
    """
    ret = {}
    for key, value in d1.items():
        ret[key] = value
    for key, value in d2.items():
        if key in ret.keys():
            if behavior is None:
                raise ValueError("Conflicting keys between the two dictionaries on key '%s'" % key)
            elif behavior == 'first':
                continue
            elif behavior == 'add':
                ret[key] += value
        ret[key] = value
    return ret


def _check_comparators(comparators):
    """
    Raises errors if comparators is not formatted correctly
    """
    if comparators is None:
        return
    if not is_list(comparators):
        raise TypeError("Comparators must be a list of lists")
    for comp in comparators:
        if not is_list(comp) or len(comp) < 3:
            raise TypeError("Every element in comparators must be a list of at least size 3")
        if not isinstance(comp[0], type) or not isinstance(comp[1], type):
            raise TypeError("Every element in comparators must have the first two elements be types")


def _cmp_objects(e1, e2, ordered, typeless_lists, remove_used, comparators):
    """
    Helper method for cmp_lists and cmp_dicts. Returns True if e1 and e2 are equal, False otherwise.
    """
    if comparators is not None:
        for cmp in comparators:
            if isinstance(e1, cmp[0]) and isinstance(e2, cmp[1]):
                return cmp[2](e1, e2, cmp[3:])
    if is_list(e1):
        if not is_list(e2) or (not typeless_lists and not isinstance(e2, type(e1))):
            return False
        return cmp_list(e1, e2, ordered=ordered, typeless_lists=typeless_lists, remove_used=remove_used,
                        comparators=comparators)
    if isinstance(e1, dict):
        return cmp_dict(e1, e2, ordered=ordered, typeless_lists=typeless_lists, comparators=comparators)
    return e1 == e2


def _find_object(o, l, checked, ordered, typeless_lists, remove_used, comparators):
    """
    Goes through l checking if o is present in an index not in checked. Returns True if found, False otherwise. Checked
        will be updated with the index of the element o if it is found
    """
    for i in [i for i in range(len(l)) if i not in checked]:
        if _cmp_objects(o, l[i], ordered, typeless_lists, remove_used, comparators):
            checked.append(i)
            return True
    return False
