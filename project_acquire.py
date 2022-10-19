#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
from env import github_token, github_username
import os
import json
import pandas as pd
import time
import csv
from typing import Dict, List, Optional, Union, cast


# In[ ]:


'''extra a repo list 
if the repo list exist in local file, we open that file
if the repo list csv is not existed, we extract it use the url and put it into csv'''
if not os.path.isfile("repo.csv"):

    repos = []
    for i in range(1, 101):
        url = f'https://github.com/search?o=desc&p={i}&q=stars%3A%3E0&s=stars&type=Repositories'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        for element in soup.find_all('a', class_='v-align-middle'):
            repos.append(element.text)

        time.sleep(10)
        print(f'Getting page {i} of 100', end='')

    with open('repo.csv', 'w') as createfile:
        wr = csv.writer(createfile, quoting=csv.QUOTE_ALL)
        wr.writerow(repos)


# In[ ]:


'''with the local csv file that has the repo,
read the file and extra a list of name and equal it to REPOS
this is an environment set up for the prepare and aquire'''
results = []
with open('repo.csv', newline='') as inputfile:
    results = list(csv.reader(inputfile))
                
REPOS = [item for sublist in results for item in sublist]


# In[8]:


"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


# In[9]:


'''use the url we have to create a function that will extra the response_date from json file'''
def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


# In[10]:


'''use the api to extra repo that will give as what repo language it is for exploration use'''
def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


# In[11]:


''' create a function with an api that will extra the repo contents '''
def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


# In[12]:


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


# In[13]:


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


# In[14]:


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


# In[ ]:


'''create a local file for data.json'''
if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)


# In[ ]:





# In[ ]:




