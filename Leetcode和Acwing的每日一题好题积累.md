# Leetcode和Acwing的好题

## 题目1：变位词组（7.18leetcode）
链接：[https://leetcode-cn.com/problems/group-anagrams-lcci/](https://leetcode-cn.com/problems/group-anagrams-lcci/)

### 题目叙述：
编写一种方法，对字符串数组进行排序，将所有变位词组合在一起。变位词是指字母相同，但排列不同的字符串。

注意：本题相对原题稍作修改

示例:

输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
说明：

所有输入均为小写字母。
不考虑答案输出的顺序。

### 思路
字母相同而排序不同，则直接按升序排序后，完全相同，然后进行hash table计数（记住在未排序前的索引即可）

### 个人cpp代码
	class Solution {
	public:
	    vector<vector<string>> groupAnagrams(vector<string>& strs) {
	        vector<string>copy_strs = strs;
	        int n=strs.size();
	        int i;
	        map<string,vector<int>> counter_map;
	        vector<int>init_vector;
	        for(i=0;i<n;i++)
	        {
	            sort(strs[i].begin(),strs[i].end());
	            //cout<<strs[i]<<endl;
	            //cout<<(counter_map.find(strs[i])==counter_map.end())<<endl;
	            if(counter_map.find(strs[i])==counter_map.end())
	            {
	                if(i!=0)init_vector.pop_back();
	                init_vector.push_back(i);
	                counter_map.insert(pair<string,vector<int>>(strs[i],init_vector));
	            }
	            else
	            {
	                counter_map[strs[i]].push_back(i);
	            }
	        }
	        //cout<<counter_map["abt"][0];
	        vector<vector<string>>result_group;
	        vector<string> strv1;
	        for(map<string,vector<int>>::iterator it=counter_map.begin();it!=counter_map.end();it++)
	        {
	            for(int j=0;j<it->second.size();j++)
	            {
	                int index;
	                index = it->second[j];
	                strv1.push_back(copy_strs[index]);
	            }
	            result_group.push_back(strv1);
	            strv1.clear();
	        }

        return result_group;
    }
};