# -*- coding: utf-8 -*-
"""
@Created on 2018/6/3 17:06

@author: ZhifengFang
"""

'''
# 排列数组删除重复项
def removeDuplicates(self, nums):
    if len(nums) <= 1:
        return len(nums)
    i = 1
    while len(nums) != i:
        if nums[i] == nums[i - 1]:
            del nums[i]
            i -= 1
        i += 1
    return len(nums)


# 买卖股票最佳时机2
def maxProfit(prices):
    max = 0
    if len(prices) <= 1:
        return 0
    for i in range(len(prices) - 1):
        if prices[i] < prices[i + 1]:
            max += prices[i + 1] - prices[i]
    return max


# 旋转数组
def rotate(nums, k):
    if len(nums) > 1:
        k = k % len(nums)
        if k != 0:
            temp = nums[-k:]
            nums[k:] = nums[:len(nums) - k]
            nums[0:k] = temp


# 判断数组中是否有重复元素
def containsDuplicate(nums):
    # if len(nums)>len(set(nums)):
    #     return True
    # return False
    for num in nums:
        if nums.count(num) > 1:
            return True
    return False


# 获得里面只出现一次的数字
def singleNumber(nums):
    numCounts = {}
    result = []
    for num in nums:
        numCounts[num] = numCounts.get(num, 0) + 1
    for key in numCounts.keys():
        if numCounts.get(key) == 1:
            result.append(key)
            break
    return result[0]


# 两个数组的交集 II
def intersect(nums1, nums2):
    if len(nums2) < len(nums1):
        nums1, nums2 = nums2, nums1
    newNums = []
    i = 0
    while i < len(nums1):
        j = 0
        while j < len(nums2):
            if nums1[i] == nums2[j]:
                newNums.append(nums2[j])
                del nums1[i], nums2[j]
                i -= 1
                j -= 1
                break
            j += 1
        i += 1
    return newNums


# 加1
def plusOne(digits):
    strDigits = ''
    for example in digits:
        strDigits += str(example)
    strDigits = int(strDigits) + 1
    listDigits = [int(str) for str in str(strDigits)]
    return listDigits


# 移动0
def moveZeroes(nums):
    # for i in range(len(nums)):
    i = 0
    zeroesCount = 0
    while i + zeroesCount < len(nums):
        if nums[i] == 0:
            nums[i:] = nums[i + 1:] + [0]
            i -= 1
            zeroesCount += 1
        i += 1
    return nums


# 两数和
def twoSum(nums, target):
    d = {}
    for x in range(len(nums)):
        a = target - nums[x]
        if nums[x] in d:
            return d[nums[x]], x
        else:
            d[a] = x


nums = [3, 2, 4]
target = 6


def isXT(strs):
    strSet = set(strs)
    for s in strSet:
        if s != ".":
            if strs.count(s) > 1:
                return False
    return True


# 有效的数独
def isValidSudoku(board):
    for i in range(9):
        boardLie = [example[i] for example in board]
        key1 = int(i / 3) * 3 + 1
        key2 = 1 + (i % 3) * 3
        boardGe = [board[key1 - 1][key2 - 1], board[key1 - 1][key2], board[key1 - 1][key2 + 1],
                   board[key1][key2 - 1], board[key1][key2], board[key1][key2 + 1],
                   board[key1 + 1][key2 - 1], board[key1 + 1][key2], board[key1 + 1][key2 + 1]]
        if isXT(board[i]) == False:
            return False
        if isXT(boardLie) == False:
            return False
        if isXT(boardGe) == False:
            return False
    return True


board = [[".", ".", "4", ".", ".", ".", "6", "3", "."],
         [".", ".", ".", ".", ".", ".", ".", ".", "."],
         ["5", ".", ".", ".", ".", ".", ".", "9", "."],
         [".", ".", ".", "5", "6", ".", ".", ".", "."],
         ["4", ".", "3", ".", ".", ".", ".", ".", "1"],
         [".", ".", ".", "7", ".", ".", ".", ".", "."],
         [".", ".", ".", "5", ".", ".", ".", ".", "."],
         [".", ".", ".", ".", ".", ".", ".", ".", "."],
         [".", ".", ".", ".", ".", ".", ".", ".", "."]]


# 旋转图像
def rotate(matrix):
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        matrix[i].reverse()


ma = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
rotate(ma)


######################################################################

# 反转字符串
def reverseString(s):
    return s[::-1]


# 颠倒数字
def reverse(x):
    if x < 0:
        flag = -2 ** 31
        result = -1 * int(str(x)[1:][::-1])
        if result < flag:
            return 0
        else:
            return result
    else:
        flag = 2 ** 31 - 1
        result = int(str(x)[::-1])
        if result > flag:
            return 0
        else:
            return result


# 字符串中的第一个唯一字符
def firstUniqChar(s):
    d = {}
    for i in range(len(s)):
        d[s[i]] = d.get(s[i], 0) + 1
    for i in range(len(s)):
        if d[s[i]] == 1:
            return i
    return -1


# 有效的字母异位词
def isAnagram(s, t):
    if len(t) != len(s):
        return False
    if len(set(t)) != len(set(s)):
        return False
    for ex in set(t):
        if s.count(ex) != t.count(ex):
            return False
    return True


# 验证回文字符串
def isPalindrome(s):
    import re
    s = s.lower()
    newS = re.sub(r'[^A-Za-z0-9]', "", s)
    if newS[::-1] == newS:
        return True
    else:
        return False


# 字符串转整数（atoi）
def myAtoi(str):
    import re
    if re.match('\s+', str) != None:
        a, b = re.match('\s+', str).span()
        str = str[b:]
    if str == '':
        return 0
    flag = True
    if str[0] == '-':
        str = str[1:]
        flag = False
    elif str[0] == '+':
        str = str[1:]
    if re.match('\d+', str) != None:
        a, b = re.match('\d+', str).span()
        str = str[a:b]
        if flag == True:
            if int(str) > 2 ** 31 - 1:
                return 2 ** 31 - 1
            return int(str)
        else:
            if -1 * int(str) < -2 ** 31:
                return -2 ** 31
            return -1 * int(str)
    else:
        return 0


# 实现 strStr() 函数。
def strStr(haystack, needle):
    if len(needle) == 0:
        return 0
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


# 数数并说
def countAndSay(n):
    keyStr = '1'  # 从1开始
    for i in range(n - 1):  # 循环n次
        newStr = ""  # 存储新的字符
        strList = []  # 循环一个字符串下来获取每个字符的个数
        sList = []  # 相同字符的个数
        flag = True
        for j in range(len(keyStr) - 1):  # 循环字符的长度减一 flag表示默认最后一个字符和前面一个字符不同，
            sList.append(keyStr[j])
            sList.append(1)
            if keyStr[j] == keyStr[j + 1]:  # 如果当前位置的字符和下一个位置的字符相同
                sList[1] += 1
                flag = False
            else:
                strList.append(sList)  # 不同的话 将上一个字符的情况存储进列表
                sList = []
                flag = True
        if flag:  # 如果最后一个字符和前一个字符不同，则将字符情况加入
            strList.append([keyStr[-1], 1])
        else:
            if sList != []:  # 最后一串相同字符加入列表
                strList.append(sList)
        for k in range(len(strList)):  # 将列表的字符按顺序取出
            newStr = newStr + '' + str(strList[k][1])
            newStr = newStr + '' + strList[k][0]
        keyStr = newStr
    return keyStr


# 最长公共前缀
def longestCommonPrefix(strs):
    if strs == []:
        return ''
    lenKey = len(strs[0])
    key = 0
    longest = 0
    for i in range(1, len(strs)):
        if len(strs[i]) < lenKey:
            key = i
            lenKey = len(strs[i])
    for i in range(len(strs[key])):
        flag = True
        for j in range(len(strs)):
            if strs[j][i] != strs[key][i]:
                flag = False
        if flag:
            longest += 1
        else:
            break
    return strs[key][0:longest]


#################################链表###################################
# 将列表转换成链表
def stringToListNode(input):
    numbers = input
    dummyRoot = ListNode(0)
    ptr = dummyRoot
    for number in numbers:
        ptr.next = ListNode(number)  # 分别将列表中每个数转换成节点
        ptr = ptr.next
    ptr = dummyRoot.next
    return ptr


# 将链表转换成字符串
def listNodeToString(node):
    if not node:
        return "[]"
    result = ""
    while node:
        result += str(node.val) + ", "
        node = node.next
    return "[" + result[:-2] + "]"


# 链表节点
class ListNode(object):
    def __init__(self, x):
        self.val = x  # 节点值
        self.next = None


class Solution(object):
    # 删除链表中的节点
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next
        # print(listNodeToString(node))

    # 删除链表的倒数第N个节点
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        listNode = []
        while head:  # 将每个节点存放在列表中
            listNode.append(head)
            head = head.next
        if 1 <= n <= len(listNode):  # 如果n在列表个数之内的话
            n = len(listNode) - n  # n原本是倒数位置，现在赋值为正方向位置
            if n == 0:  # 如果是删除第1个位置的节点
                if len(listNode) > 1:  # 如果节点总数大于1
                    listNode[0].val = listNode[1].val  # 删除第1个位置
                    listNode[0].next = listNode[1].next
                else:
                    return None  # 因为节点一共就1个或0个，所以删除1个直接返回None
            else:
                listNode[n - 1].next = listNode[n].next  # 将该节点的上一个节点的后节点赋值为该节点的后节点，即删除该节点
        return listNode[0]

    # 反转链表
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        listNode = []
        while head:
            listNode.append(head)
            head = head.next
        if len(listNode) == 0:
            return None
        for i in range(int(len(listNode) / 2)):  # 将节点的值收尾分别调换
            listNode[i].val, listNode[len(listNode) - i - 1].val = listNode[len(listNode) - i - 1].val, listNode[i].val
        return listNode[0]

    # 合并两个有序链表
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        newList = ListNode(0)
        newList.next = l1
        prev = newList  # 获得新链表

        while l2:
            if not l1:  # 如果l1不存在，直接返回l2即可
                prev.next = l2
                break
            if l1.val > l2.val:  # 1，判断l1和l2哪个大，如果l2小，则将新节点的后面设为l2的头节点，并将头节点的后面设置为l1，反之l1小，则直接将头节点的后面设置为l1，并将节点后移
                temp = l2
                l2 = l2.next
                prev.next = temp
                temp.next = l1
                prev = prev.next  #
            else:  # 反之l2大于l1，则是l1节点向后移
                l1, prev = l1.next, l1
        return newList.next

    # 回文链表
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        listNode = []
        while head:
            listNode.append(head)
            head = head.next
        for i in range(int(len(listNode) / 2)):  # 判断两头的值是否一样大
            if listNode[i].val != listNode[len(listNode) - i - 1].val:
                return False
        return True

    # 环形链表
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return False
        p1 = p2 = head
        while p2.next and p2.next.next:  # p1走1步，p2走两步，如果在链表没走完的情况下，找到完全相同的节点，就是找到环了
            p1 = p1.next
            p2 = p2.next.next
            if p1 == p2:
                return True
        return False


head = [1, 2, 3, 4, 5]
head2 = [4, 5, 8, 9]
node = 1
s = Solution()

# print(s.deleteNode(stringToListNode(head)))  # 删除第一个位置
# print(listNodeToString(s.removeNthFromEnd(stringToListNode(head), 1)))  # 删除倒数第一个位置
# print(listNodeToString(s.reverseList(stringToListNode(head))))  # 翻转
print(listNodeToString(s.mergeTwoLists(stringToListNode(head2), stringToListNode(head))))  # 合并两个链表
# print(s.isPalindrome(stringToListNode(head)))
# print(s.hasCycle(stringToListNode(head)))

#####################################排序和搜索#######################################
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        nums1[m:m + n] = nums2[:n]
        nums1.sort()
    
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        minn=1
        maxn=n
        while True:
            mid=int((minn+maxn)/2)
            if isBadVersion(mid)==True and isBadVersion(mid+1)==True:
                maxn=mid-1
            elif isBadVersion(mid)==False and isBadVersion(mid+1)==False:
                minn=mid+1
            else:
                return mid+1

# nums1 = [4,0,0,0,0,0]
#
# m = 1
#
# nums2 = [1,2,3,5,6]
#
# n = 5
# print(Solution().merge(nums1, m, nums2, n))

##################################树####################################
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0
        leftCount = 1
        rightCount = 1
        leftCount = leftCount + self.maxDepth(root.left)
        rightCount = rightCount + self.maxDepth(root.right)
        return max(leftCount, rightCount)

    def validBST(self, root, min, max):
        if root == None:
            return True
        if root.val <= min or root.val >= max:
            return False
        return self.validBST(root.left, min, root.val) and self.validBST(root.right, root.val, max)

    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.validBST(root, -2 ** 64 + 1, 2 ** 64 - 1)

    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True
        if root.left != None and root.right != None:
            if root.left.val==root.right.val:
                return self.isSymmetric(root.left) and self.isSymmetric(root.right)
            else:
                return False
        elif root.left == None and root.right == None:
            return True
        else:
            return False
        return True

def stringToTreeNode(input):
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(',')]
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

def intToString(input):
    if input is None:
        input = 0
    return str(input)


# line = '[1,2,2,3,4,4,3]'
# root = stringToTreeNode(line)
# # ret = Solution().maxDepth(root)
# # ret = Solution().isValidBST(root)
# ret = Solution().isSymmetric(root)
# out = intToString(ret)
# print(out)'''


##################################动态规划########################################

# 动态规划的本质是递归；所以首先用暴力的方式写出第一步递归方法；递归是自顶向下，如求第n个数时，函数返回的是n-1加n-2，所以问题会回到n-1，继续求n-1下面的值
# 但是这样会有很多冗余，如当slove(n-1)和slove(n-2)里面 n-2后面所要计算的都一样 重复计算了，所以可以从1、2开始，循环用一个数组存储；

class Solution(object):
    # 爬楼梯
    def recursionClimbStairs(self, n):
        if n <= 2:
            return n
        return self.recursionClimbStairs(n - 1) + self.recursionClimbStairs(n - 2)

    def climbStairs(self, n):
        nums = [1, 2]
        if n <= 2:
            return n
        for i in range(2, n):
            nums.append(nums[i - 1] + nums[i - 2])
        return nums[len(nums) - 1]

    # 买卖股票的最佳时机
    def recursionMaxProfix(self, prices):
        if len(prices) < 2:
            return 0
        return max(prices[- 1] - min(prices[:- 1]),
                   self.recursionMaxProfix(prices[:- 1]))

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        result = [0]
        if len(prices) < 2:
            return 0
        minPrice = prices[0]
        for i in range(1, len(prices)):
            minPrice = min(minPrice, prices[i - 1])
            result.append(max(prices[i] - minPrice, result[i - 1]))
        return result[-1]

    def recursionMaxSubArray(self, idx, nums, maxSum):
        if idx < 0:
            return maxSum
        nums[idx] = max(nums[idx], nums[idx + 1] + nums[idx])
        maxSum = max(nums[idx], maxSum)
        return self.recursionMaxSubArray(idx - 1, nums, maxSum)

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(1, len(nums)):
            nums[i] = max(nums[i] + nums[i - 1], nums[i])
        return max(nums)

    # 打家劫舍
    def recursionRob(self, idx, nums):
        if idx < 0:
            return 0
        return max(nums[idx] + self.recursionRob(idx - 2, nums), self.recursionRob(idx - 1, nums))

    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        nums[1] = max(nums[0], nums[1])
        for i in range(2, len(nums)):
            nums[i] = max(nums[i] + nums[i - 2], nums[i - 1])
        return nums[len(nums) - 1]


a = [-1,-2]
# print(Solution().maxSubArray(a))

print(Solution().recursionMaxSubArray(len(a) - 2, a, a[-1]))
