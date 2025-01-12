import nodes

from comfy_execution.graph_utils import is_link

class DependencyCycleError(Exception):
    pass

class NodeInputError(Exception):
    pass

class NodeNotFoundError(Exception):
    pass

class DynamicPrompt:
    """
    DynamicPrompt类用于管理一个动态变化的提示信息结构，主要包含用户原始的提示以及在执行过程中创建的临时节点信息。
    """
    def __init__(self, original_prompt):
        """
        初始化函数，用于创建DynamicPrompt对象。

        :param original_prompt: 用户提供的原始提示信息，通常是一个字典。
        """
        # The original prompt provided by the user
        # 用户提供的原始提示信息
        self.original_prompt = original_prompt
        # Any extra pieces of the graph created during execution
        # 执行过程中创建的临时节点提示信息
        self.ephemeral_prompt = {}
        # 临时节点的父节点信息
        self.ephemeral_parents = {}
        # 临时节点的显示信息
        self.ephemeral_display = {}

    def get_node(self, node_id):
        """
        获取指定节点的信息。

        :param node_id: 节点的唯一标识符。
        :return: 返回节点的信息，如果节点不存在，则抛出NodeNotFoundError异常。
        """
        # 优先从临时节点中查找
        if node_id in self.ephemeral_prompt:
            return self.ephemeral_prompt[node_id]
        # 如果不存在临时节点中，则从原始提示信息中查找
        if node_id in self.original_prompt:
            return self.original_prompt[node_id]
        # 如果都找不到，抛出异常
        raise NodeNotFoundError(f"Node {node_id} not found")

    def has_node(self, node_id):
        """
        检查是否存在指定的节点。

        :param node_id: 节点的唯一标识符。
        :return: 布尔值，表示是否包含该节点。
        """
        return node_id in self.original_prompt or node_id in self.ephemeral_prompt

    def add_ephemeral_node(self, node_id, node_info, parent_id, display_id):
        """
        添加一个临时节点。

        :param node_id: 节点的唯一标识符。
        :param node_info: 节点的相关信息。
        :param parent_id: 节点的父节点标识符。
        :param display_id: 节点的显示标识符。
        """
        self.ephemeral_prompt[node_id] = node_info
        self.ephemeral_parents[node_id] = parent_id
        self.ephemeral_display[node_id] = display_id

    def get_real_node_id(self, node_id):
        """
        获取节点的实际ID，如果节点有父节点，则追溯到最原始的节点ID。

        :param node_id: 节点的唯一标识符。
        :return: 节点的实际ID。
        """
        while node_id in self.ephemeral_parents:
            node_id = self.ephemeral_parents[node_id]
        return node_id

    def get_parent_node_id(self, node_id):
        """
        获取节点的父节点ID。

        :param node_id: 节点的唯一标识符。
        :return: 节点的父节点ID，如果不存在父节点，则返回None。
        """
        return self.ephemeral_parents.get(node_id, None)

    def get_display_node_id(self, node_id):
        """
        获取节点的显示ID，如果节点有显示映射的话。

        :param node_id: 节点的唯一标识符。
        :return: 节点的显示ID。
        """
        while node_id in self.ephemeral_display:
            node_id = self.ephemeral_display[node_id]
        return node_id

    def all_node_ids(self):
        """
        获取所有节点的ID，包括原始提示信息中的节点和临时节点。

        :return: 包含所有节点ID的集合。
        """
        return set(self.original_prompt.keys()).union(set(self.ephemeral_prompt.keys()))

    def get_original_prompt(self):
        """
        获取原始的提示信息。

        :return: 原始的提示信息字典。
        """
        return self.original_prompt


def get_input_info(class_def, input_name, valid_inputs=None):
    """
    获取输入信息

    该函数根据类定义和输入名称，返回指定输入的类型、类别以及其他信息
    主要用于解析类的输入要求，以确定输入在该类中的角色和属性

    参数:
    - class_def: 类的定义，用于查询输入类型
    - input_name: 需要查询的输入名称

    返回:
    - input_type: 输入的类型
    - input_category: 输入的类别（如"required", "optional", "hidden"）
    - extra_info: 其他额外信息，通常是一个字典
    """
    # 获取类的输入类型定义
    valid_inputs = valid_inputs or class_def.INPUT_TYPES()
    # 初始化输入信息和类别
    input_info = None
    input_category = None
    # 检查输入名称是否在"required"类别中
    if "required" in valid_inputs and input_name in valid_inputs["required"]:
        input_category = "required"
        input_info = valid_inputs["required"][input_name]
    # 如果不在"required"中，检查是否在"optional"类别中
    elif "optional" in valid_inputs and input_name in valid_inputs["optional"]:
        input_category = "optional"
        input_info = valid_inputs["optional"][input_name]
    # 如果不在"optional"中，检查是否在"hidden"类别中
    elif "hidden" in valid_inputs and input_name in valid_inputs["hidden"]:
        input_category = "hidden"
        input_info = valid_inputs["hidden"][input_name]
    # 如果input_info仍为None，则返回None值三元组
    if input_info is None:
        return None, None, None
    # 获取输入的类型
    input_type = input_info[0]
    # 初始化额外信息，如果存在则赋值，否则为空字典
    if len(input_info) > 1:
        extra_info = input_info[1]
    else:
        extra_info = {}
    # 返回输入的类型、类别和其他信息
    return input_type, input_category, extra_info


class TopologicalSort:
    # 定义拓扑排序类，用于处理有依赖关系的节点
    def __init__(self, dynprompt):
        # 初始化方法
        # 参数 dynprompt: 动态提示对象，用于获取节点信息
        self.dynprompt = dynprompt
        # 待处理节点字典
        self.pendingNodes = {}
        # 节点被直接阻塞的数量
        self.blockCount = {} # Number of nodes this node is directly blocked by
        # 阻塞其他节点的列表
        self.blocking = {} # Which nodes are blocked by this node

    # 获取特定输入的节点信息
    # 参数 unique_id: 节点唯一标识符
    # 参数 input_name: 输入名称
    # 返回: 指定节点的输入信息
    def get_input_info(self, unique_id, input_name):
        class_type = self.dynprompt.get_node(unique_id)["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        return get_input_info(class_def, input_name)

    # 为指定输入创建强链接
    # 参数 to_node_id: 目标节点唯一标识符
    # 参数 to_input: 需要建立强链接的输入名称
    def make_input_strong_link(self, to_node_id, to_input):
        # 获取目标节点的输入字典
        inputs = self.dynprompt.get_node(to_node_id)["inputs"]
        # 检查目标节点是否具有所需的输入
        if to_input not in inputs:
            raise NodeInputError(f"Node {to_node_id} says it needs input {to_input}, but there is no input to that node at all")
        # 获取所需输入的值
        value = inputs[to_input]
        # 检查输入值是否为链接类型
        if not is_link(value):
            raise NodeInputError(f"Node {to_node_id} says it needs input {to_input}, but that value is a constant")
        # 解包链接值，获取源节点ID和源节点输出插槽
        from_node_id, from_socket = value
        # 增加强链接，将源节点和目标节点连接起来
        self.add_strong_link(from_node_id, from_socket, to_node_id)

    # 在两个节点之间添加强链接
    # 参数 from_node_id: 源节点唯一标识符
    # 参数 from_socket: 源节点插槽标识符
    # 参数 to_node_id: 目标节点唯一标识符
    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        # 添加源节点ID到图中
        if not self.is_cached(from_node_id):
            self.add_node(from_node_id)
            # 如果目标节点ID不在源节点的阻塞列表中，则初始化阻塞记录并增加目标节点的阻塞计数
            if to_node_id not in self.blocking[from_node_id]:
                self.blocking[from_node_id][to_node_id] = {}
                self.blockCount[to_node_id] += 1
            # 在源节点的阻塞列表中，将特定源端口标记为被目标节点阻塞
            self.blocking[from_node_id][to_node_id][from_socket] = True

    # 将节点添加到待处理列表，并更新其依赖信息
    # 参数 unique_id: 节点唯一标识符
    def add_node(self, node_unique_id, include_lazy=False, subgraph_nodes=None):
        # 检查当前节点是否已经在待处理队列中，避免重复处理
        node_ids = [node_unique_id]
        links = []

        while len(node_ids) > 0:
            unique_id = node_ids.pop()
            if unique_id in self.pendingNodes:
                continue

            # 将当前节点标记为待处理状态
            self.pendingNodes[unique_id] = True
            # 初始化当前节点的阻塞计数为0
            self.blockCount[unique_id] = 0
            # 初始化当前节点的阻塞依赖字典
            self.blocking[unique_id] = {}

            # 获取当前节点的输入信息
            inputs = self.dynprompt.get_node(unique_id)["inputs"]
            for input_name in inputs:
                # 获取输入值
                value = inputs[input_name]
                # 判断输入值是否为链接类型
                if is_link(value):
                    # 解析链接，获取上游节点ID和输出端口
                    from_node_id, from_socket = value
                    # 如果指定了子图范围且上游节点不在该范围内，则跳过处理
                    if subgraph_nodes is not None and from_node_id not in subgraph_nodes:
                        continue
                    # 获取当前节点输入的类型、分类和附加信息
                    input_type, input_category, input_info = self.get_input_info(unique_id, input_name)
                    # 判断当前输入是否为延迟执行类型
                    is_lazy = input_info is not None and "lazy" in input_info and input_info["lazy"]
                    # 根据是否包含延迟执行来决定是否添加强链接
                    if (include_lazy or not is_lazy) and not self.is_cached(from_node_id):
                        node_ids.append(from_node_id)
                        # 添加强链接，表示当前节点依赖上游节点的特定输出
                        links.append((from_node_id, from_socket, unique_id))

        for link in links:
            self.add_strong_link(*link)

    def is_cached(self, node_id):
        return False

    # 获取未被阻塞且准备就绪的节点
    # 返回: 未被阻塞的节点标识符列表
    def get_ready_nodes(self):
        return [node_id for node_id in self.pendingNodes if self.blockCount[node_id] == 0]

    # 从待处理列表中移除节点，并更新它所阻塞的节点的依赖计数
    # 参数 unique_id: 节点唯一标识符
    def pop_node(self, unique_id):
        # 从待处理节点字典中移除已经处理完毕的节点
        del self.pendingNodes[unique_id]

        # 遍历该节点所阻塞的所有节点，并减少它们的阻塞计数
        for blocked_node_id in self.blocking[unique_id]:
            self.blockCount[blocked_node_id] -= 1

        # 从阻塞关系字典中移除已经处理完毕的节点的信息
        del self.blocking[unique_id]

    # 检查待处理列表是否为空
    # 返回: 表示待处理列表是否为空的布尔值
    def is_empty(self):
        return len(self.pendingNodes) == 0


class ExecutionList(TopologicalSort):
    """
    ExecutionList implements a topological dissolve of the graph. After a node is staged for execution,
    it can still be returned to the graph after having further dependencies added.
    """
    def __init__(self, dynprompt, output_cache):
        super().__init__(dynprompt)
        self.output_cache = output_cache
        self.staged_node_id = None

    def is_cached(self, node_id):
        return self.output_cache.get(node_id) is not None

    def stage_node_execution(self):
        assert self.staged_node_id is None
        if self.is_empty():
            return None, None, None
        available = self.get_ready_nodes()
        if len(available) == 0:
            cycled_nodes = self.get_nodes_in_cycle()
            # Because cycles composed entirely of static nodes are caught during initial validation,
            # we will 'blame' the first node in the cycle that is not a static node.
            blamed_node = cycled_nodes[0]
            for node_id in cycled_nodes:
                display_node_id = self.dynprompt.get_display_node_id(node_id)
                if display_node_id != node_id:
                    blamed_node = display_node_id
                    break
            ex = DependencyCycleError("Dependency cycle detected")
            error_details = {
                "node_id": blamed_node,
                "exception_message": str(ex),
                "exception_type": "graph.DependencyCycleError",
                "traceback": [],
                "current_inputs": []
            }
            return None, error_details, ex

        self.staged_node_id = self.ux_friendly_pick_node(available)
        return self.staged_node_id, None, None

    def ux_friendly_pick_node(self, node_list):
        # If an output node is available, do that first.
        # Technically this has no effect on the overall length of execution, but it feels better as a user
        # for a PreviewImage to display a result as soon as it can
        # Some other heuristics could probably be used here to improve the UX further.
        def is_output(node_id):
            class_type = self.dynprompt.get_node(node_id)["class_type"]
            class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
            if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                return True
            return False

        for node_id in node_list:
            if is_output(node_id):
                return node_id

        #This should handle the VAEDecode -> preview case
        for node_id in node_list:
            for blocked_node_id in self.blocking[node_id]:
                if is_output(blocked_node_id):
                    return node_id

        #This should handle the VAELoader -> VAEDecode -> preview case
        for node_id in node_list:
            for blocked_node_id in self.blocking[node_id]:
                for blocked_node_id1 in self.blocking[blocked_node_id]:
                    if is_output(blocked_node_id1):
                        return node_id

        #TODO: this function should be improved
        return node_list[0]

    def unstage_node_execution(self):
        assert self.staged_node_id is not None
        self.staged_node_id = None

    def complete_node_execution(self):
        node_id = self.staged_node_id
        self.pop_node(node_id)
        self.staged_node_id = None

    def get_nodes_in_cycle(self):
        # We'll dissolve the graph in reverse topological order to leave only the nodes in the cycle.
        # We're skipping some of the performance optimizations from the original TopologicalSort to keep
        # the code simple (and because having a cycle in the first place is a catastrophic error)
        # 初始化一个字典，用于记录每个节点被哪些节点阻塞
        blocked_by = { node_id: {} for node_id in self.pendingNodes }
        # 遍历当前阻塞关系中的所有节点
        for from_node_id in self.blocking:
            # 遍历从某个节点阻塞的其他所有节点
            for to_node_id in self.blocking[from_node_id]:
                # 如果存在任何True的阻塞状态
                if True in self.blocking[from_node_id][to_node_id].values():
                    # 更新blocked_by字典，记录to_node_id被from_node_id阻塞
                    blocked_by[to_node_id][from_node_id] = True

        # 找出没有阻塞其他任何节点的节点
        to_remove = [node_id for node_id in blocked_by if len(blocked_by[node_id]) == 0]

        # 逐个移除那些没有阻塞任何节点的节点，以更新blocked_by字典
        while len(to_remove) > 0:
            # 遍历需要移除的节点
            for node_id in to_remove:
                # 更新其他节点的阻塞状态，移除对即将被删除的节点的引用
                for to_node_id in blocked_by:
                    if node_id in blocked_by[to_node_id]:
                        del blocked_by[to_node_id][node_id]
                # 从blocked_by字典中删除当前节点
                del blocked_by[node_id]
            # 重新找出没有阻塞其他任何节点的节点
            to_remove = [node_id for node_id in blocked_by if len(blocked_by[node_id]) == 0]

        # 返回所有剩余节点的列表，这些节点至少阻塞了另一个节点
        return list(blocked_by.keys())

class ExecutionBlocker:
    """
    Return this from a node and any users will be blocked with the given error message.
    If the message is None, execution will be blocked silently instead.
    Generally, you should avoid using this functionality unless absolutely necessary. Whenever it's
    possible, a lazy input will be more efficient and have a better user experience.
    This functionality is useful in two cases:
    1. You want to conditionally prevent an output node from executing. (Particularly a built-in node
       like SaveImage. For your own output nodes, I would recommend just adding a BOOL input and using
       lazy evaluation to let it conditionally disable itself.)
    2. You have a node with multiple possible outputs, some of which are invalid and should not be used.
       (I would recommend not making nodes like this in the future -- instead, make multiple nodes with
       different outputs. Unfortunately, there are several popular existing nodes using this pattern.)
    """
    def __init__(self, message):
        self.message = message

