# coding=utf-8

# empty interpolation interface class
class IInterpolation(object):
    def __init__(*args, **kwargs):
        pass

    def get_shape_of_nested_list(self, nested_list):
        """gets the shape of a nested list it is structured symmetrically

        """
        part_of_list = nested_list
        shape = []
        while isinstance(part_of_list, list):
            shape.append(len(part_of_list))
            part_of_list = part_of_list[0]
        return shape
