# -*- coding: utf-8 -*-
"""
@Created on 2018/6/3 17:06

@author: ZhifengFang
"""


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
    # nums = nums[-k:] + nums[:k + 1]
    # print(nums)
    if len(nums) > 1:
        k = k % len(nums)
        if k != 0:
            temp = nums[-k:]
            nums[k:] = nums[:len(nums) - k]
            nums[0:k] = temp
    print(nums)


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


# print(intersect([9],[7,8,3,9,0,0,9,1,5]))


# 加1
def plusOne(digits):
    strDigits = ''
    for example in digits:
        strDigits += str(example)
    strDigits = int(strDigits) + 1
    listDigits = [int(str) for str in str(strDigits)]
    return listDigits


# print(plusOne([1, 2, 3]))


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


# print(twoSum(nums, target))

def isXT(strs):
    for i in range(len(strs)):
        if strs[i] != ".":
            if strs.count(strs[i]) > 1:
                return False
    return True


def isValidSudoku(board):
    flag = False
    for i in range(9):
        boardLie = []
        for j in range(9):
            boardLie.append(board[i][j])
            k = j % 3
            boardGe = board[i][j]
        if isXT(board[i]) == False:
            return False
        if isXT(boardLie) == False:
            return False

    return flag


board = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"]
]


# print(isXT(board[0]))
# 反转字符串
def reverseString(s):
    return s[::-1]


# print(('123'))
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

# print(reverse(-1563847412))
def firstUniqChar(s):
    d={}
    for i in range(len(s)):
        d[s[i]] = d.get(s[i], 0) + 1
    for i in range(len(s)):
        if d[s[i]]==1:
            return i
    return -1


def strStr( haystack, needle):
    if len(needle) == 0:
        return 0
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1

print(strStr("a","a"))