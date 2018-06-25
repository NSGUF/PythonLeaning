# -*- coding: utf-8 -*-
"""
@Created on 2018/6/20 20:31

@author: ZhifengFang
"""


# 定义二叉树
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    # 前序遍历
    def solvePre(self, root, result):
        if root == None:
            return []
        result.append(int(root.val))  # 先添加根节点
        self.solvePre(root.left, result)  # 再添加左子树
        self.solvePre(root.right, result)  # 再添加又子树
        return result

    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        return self.solvePre(root, [])

    # 中序遍历
    def solveIno(self, root, result):
        if root == None:
            return []
        self.solveIno(root.left, result)  # 先遍历左子树
        result.append(int(root.val))
        self.solveIno(root.right, result)  # 再遍历右子树
        return result

    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        return self.solveIno(root, [])

    # 后序遍历
    def solvePos(self, root, result):
        if root == None:
            return []
        self.solvePos(root.left, result)  # 先访问左子树
        self.solvePos(root.right, result)  # 在访问右子树
        result.append(int(root.val))
        return result

    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        return self.solvePos(root, [])

    # 层次遍历
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root == None:
            return []
        queue = [root]  # 通过list存储
        result = []
        index = 0
        while True:
            start = index  # 该层的节点在list中的开始位置
            end = len(queue)  # 该层的节点在list中的最后位置
            block = []  # 存储该层的数据
            for i in range(start, end):
                block.append(queue[i].val)
                if queue[i].left != None:
                    queue.append(queue[i].left)
                if queue[i].right != None:
                    queue.append(queue[i].right)
                index += 1
            if start >= end:  # 如果list中的元素被循环完了，即没有添加新的节点
                break
            result.append(block)
        return result

    # 最大深度
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        leftCount = 1  # 左子树的深度
        rightCount = 1  # 右子树的深度
        leftCount = leftCount + self.maxDepth(root.left)
        rightCount = rightCount + self.maxDepth(root.right)
        return max(leftCount, rightCount)  # 选深度大的

    # 对称二叉树
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True
        index = 0
        queue = [root]
        while True:  # 获取每层的数据存入list，如果list翻转后和list不一样 则表示不是对称的
            start = index
            end = len(queue)
            block = []
            for i in range(start, end):
                if queue[i] == None:
                    block.append(' ')  # 这是为了确定某个地方为空的，里面的数据只有不为数字就行
                else:
                    block.append(queue[i].val)
                    queue.append(queue[i].left)
                    queue.append(queue[i].right)
                index += 1

            if block[::-1] != block:
                return False
            if index >= len(queue):
                break
            if block.count(' ') == len(block):  # 当该层的数据都是空的，则跳出
                break
        return True

    def solveHasPathSum(self, root, sumAll, sum, flag):  # 使用深度搜索
        if root:
            sumAll += root.val
            if root.left or root.right:  # 当该节点不是叶子节点时
                flag = self.solveHasPathSum(root.left, sumAll, sum, flag) | self.solveHasPathSum(root.right, sumAll,
                                                                                                 sum, flag)
            else:
                if sumAll == sum:
                    return True
        return flag

    # 路径总和
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        return self.solveHasPathSum(root, 0, sum, False)

    # 100. 相同的树
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p == None and q == None:
            return True
        if p != None and q != None and p.val == q.val:  # 如果pq相同，则判断他们的子节点是否也相同
            return self.isSameTree(p.left, q.left) & self.isSameTree(p.right, q.right)
        return False

    # 从前序与中序遍历序列构造二叉树
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if preorder == []:
            return None
        root = TreeNode(preorder[0])  # 前序遍历的第一个节点是父节点
        index = inorder.index(preorder[0])  # 父节点的左边是左子树的，右边是右子树的
        root.left = self.buildTree(preorder[1:index + 1], inorder[:index])
        root.right = self.buildTree(preorder[index + 1:], inorder[index + 1:])
        return root

    # 从中序与后序遍历序列构造二叉树
    def buildTree(self, inorder, postorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if postorder == []:
            return None
        root = TreeNode(postorder[-1])  # 后序遍历的最后一个节点是父节点
        index = inorder.index(postorder[-1])  # 父节点的左边是左子树的，右边是右子树的
        root.left = self.buildTree(inorder[:index], postorder[:index])
        root.right = self.buildTree(inorder[index + 1:], postorder[index:-1])
        return root

    # 每个节点的右向指针 II
    def connect(self, root):
        if root != None:
            index = 0
            queue = [root]
            while True:
                start = index
                end = len(queue)
                for i in range(start, end):
                    if i != end - 1:  # 如果该层的最后一个节点，则指向空，否则指向后一个节点
                        queue[i].next = queue[i + 1]
                    else:
                        queue[i].next = None
                    if queue[i].left:  # 节点存在则添加，否则不添加
                        queue.append(queue[i].left)
                    if queue[i].right:
                        queue.append(queue[i].right)
                    index += 1
                if index >= len(queue):
                    break

    def isContent(self, root, num):  # 判断该树是否有该节点
        if root == None:
            return False
        if root.val == num.val:
            return True
        return self.isContent(root.left, num) | self.isContent(root.right, num)

    # 二叉树的最近公共祖先
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root == None or root.val == p.val or root.val == q.val:  # 如果根节点是p或q则该节点是lca
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left != None else right

    # 序列化
    def serialize(self, root):
        """Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        """
        if root == None:
            return []
        queue = [root]
        result = []
        index = 0
        nullIndex = 0
        while True:
            start = index
            end = len(queue)
            flag = True
            for i in range(start, end):
                if queue[i] != None:
                    flag = False
                    continue
            if flag == False:
                for i in range(start, end):
                    if queue[i] == None:
                        result.append('null')
                    else:
                        result.append(queue[i].val)
                        nullIndex = i
                        if queue[i].left != None:
                            queue.append(queue[i].left)
                        else:
                            queue.append(None)
                        if queue[i].right != None:
                            queue.append(queue[i].right)
                        else:
                            queue.append(None)
                    index += 1
            else:
                break

        return result[:nullIndex + 1]

    # 反序列化
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        :type data: str
        :rtype: TreeNode
        """
        inputValues = data
        if data == []:
            return None
        root = TreeNode((inputValues[0]))
        nodeQueue = [root]
        front = 0
        index = 1
        while index < len(inputValues):
            node = nodeQueue[front]
            front = front + 1

            item = inputValues[index]
            index = index + 1
            if item != "null":
                leftNumber = int(item)
                node.left = TreeNode(leftNumber)
                nodeQueue.append(node.left)

            if index >= len(inputValues):
                break

            item = inputValues[index]
            index = index + 1
            if item != "null":
                rightNumber = int(item)
                node.right = TreeNode(rightNumber)
                nodeQueue.append(node.right)
        return root


def stringToTreeNode(input):
    inputValues = input
    root = TreeNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = TreeNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = TreeNode(rightNumber)
            nodeQueue.append(node.right)
    return root


if __name__ == '__main__':
    line = [5, 2, 3, 'null', 'null', 2, 4, 3, 1]

    root = stringToTreeNode(line);
    # ret = Solution().levelOrder(root)

    ret = Solution().serialize(root)
    print(ret)
