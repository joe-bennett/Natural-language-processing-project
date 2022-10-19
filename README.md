# Natural Language Processing - Repository main programming language predictor


## Project description:

- We were able to programatically scrape some of the most popular repositories on Github, and from them collect the repository name, main programming language used, and the content of the README file from each. 

- All the data collected was compiled into a DataFrame with the anticipation that we will be able to glean from it what the main programming language is of the repository via Natural Language Processing methods on the corresponding README file content.

- While the data we collected is static, the Github repositories are dynamic and ever-changing. Therefore is important to note the data was collected from these repositories on the morning of October 18, 2022.

## Project goal

- The stated goal of this project is to predict the main progamming language of a repository on Github based on the commensurate README file contents.

## Project Dictionary

| Target      | Description  |
| ------------- |:-------------:|
| language      | The target variable |

----------------------------------------------------
| Features      | Description  |
| ------------- |:-------------:|
| repo    | The username and repo name |
| readme_contents     | The original readme contents     |
| stemmed | The stemmed readme content    |
| lemmatized | The lemmatized readme content      |
| clean_tokens | The cleaned readme content      |
| word_count_simple | The count of space in lemmatized content      |
| word count | The word count in lemmatized content      |
|unique_count|unique word count|
|non_single_count|Count of words appearing more then once|
|percent_unique|Percentage of unique words|
|percent_repeat|Percentage of repeated words|
|percent_one_word|Percentage of single time appearance in all words|
|percent_non_single|Percentage of words that appeared more than once|


## Project plan

- data science across all domains can usually be generalized as the following steps. We used this as our framework to make our plan.

    - Planning- writing out a timeline of actionable items, when the MVP will be finished and how to know when it is complete, formulate initial questions to ask the data.

    - Acquisition- Gather data via programatically scraping README.md files from popular Github repositories and bring all necessary data into python enviroment.

    - Preparation- this is blended with acquisition where we clean and tidy the data to make it apropriate for natural language programming techniques, and split into train, validate, and test.

    - Exploration/Pre-processing- we create visualizations and answer the questions we initially set out to answer to select and engineer features that impact the target variable.

    - Modeling- based on what we learn in the exploration of the data I will select the useful features and feed into different  models with different hyperparameters and evaluate performance of each to select our best perfomoing model.

    - Delivery- create a final report and Google slides summary that succintly summarizes what we did, why we did it, what we learned,and any relavent conclusions.
    

## Initial Questions/Hypotheses

### Question 1
- What are the most common words in READMEs?
### Question 2
- Does the length of the README vary by programming language? If not whether the bigram different per language?
### Question 3
- Do different programming languages use a different number of unique words? if yes, whether there can find some corelation between the languages?
#### Hypothesis
- H0= There is no difference between unique words and language
- H1= There is difference between unique words and language
- Alpha =0.05

### Question 4
- whether the mean of percentage_one_word in python is greater than the mean of percentage one word in JavaScript
#### Hypothesis
- H0: Mean of &one word in python <= Mean of &one word in javascript
- Ha: Mean of &one word in python > Mean of &one word in javascript
- Alpha =0.05


## Conclusions & Google slides summary link

### Summary link

- https://docs.google.com/presentation/d/16OJMqrVQVK5rhsMfJ8tBbMo1kPForpWMpO9zs0waHx8/edit?usp=sharing

### Conclusion
- The decision tree bags of word has best outcome in our training model
- We used that, and was able to achived 50% accuracy
- We created a function to predict the outcome
- we successfully predict the language based on a random (not so random), readme.
- although Java is using a lot unique word, but that proportion to their readme the python use more unique word
- The amount of unique word has contribute to the language
- Each Language has unique word the distinguish to other language
- The tope ten most used words are: use,install,using,run,file,build,code,version,support,project

### Recommendation
- we found that unique word can help to predict the language
- We believe accuracy becomes uncertain when we try to predict more language, so try to predict three most commonly used language

### Next Step
- we found repeat rows in our dataset, we believe there are errors in our coding, we need to find out a way to use api scraping without capture same information
- Instead of five languages, we will reduce it to three, so we can increase our accuracy


## How to repeat this work

- You will need to pull down all files in this repository into the working directory in order for the workbook to run top down without issue.

- You will need all the libraries listed at the top of the final workbook installed. Simply PIP install whatever library you do not have as needed.

