# COSC311 Lab 1

by Justin Ventura

This file just contains the text answers to the questions in the lab which were already answered in the notebook (.ipynb) included in this directory.

## Text Answers

### How can we visualize the “type” of each song? What styles and what types of artists tend to play each song?

Well, to visualize each song, one great option could be to use a radar chart.  This is useful because the radar could have every keyword in the database be a variable on the chart, then take each artist who worked on the song, take their kewords, and add each keyword to its respective axis on the chart. The flaw with this, however, is that the more keywords in the database, the harder it becomes to read the data.  So this would be best to use with limited keywords (in this specific example, there are a reasonable amount of keywords to create a readable radar chart).  If the number of possible keywords  was arbitrarily large for a song with an arbitrarily large and a uniformly distributed pool of artists, we could consider using a bar chart.  This way, you could consider each 'bucket' to be a space for a keyword, then the greater the y-value (height) the bar in the bucket (discrete x-point), the more that keyword is associated with the song in question.  In both of these cases, we could implement some sort of database with specific 'types' of songs, which then are assigned a specific shape of radar chart/histogram.  Then each visualization could be compared against the predetermined 'types.'  Then if there is no exact match, calculate the error against each type, and whichever has minimal error could be a likely candidate to give it a type.  You could also come up with a machine learning algorithm that, very generally (since I am still very new to machine learning at this time), could be trained to learn types of songs and their respective graphs.  Then test it on songs which it has never 'seen' before to access its accuracy.

### For a specific example, what visualization can answer the question: What style of song is “Simple Twist of Fate”? Does it have a style or does it “defy genre”?

Any of the previously mentioned could be used for this.  Which one is the best and most readable?  I don't know yet, however after more experience with problems like this, I think I would be able to give a good answer to that question in due time.  

### Final questions:

- (a) Sketch by hand or digitally an example of your idea, using the data you entered above.

Please see the last python cell in the notebook for this answer.

- (b) Investigate how you might get Python to automate the production of your idea. What libraries can you find that might be useful?

As seen above, I used 'matplotlib.pyplot'.  There are some other libraries such as pandas, numpy, keras, sci-kit, etc.  

- (c) Finally, write down one strategy to try answering the above questions about song-style quantitatively. Why? If you’re, say, Spotify, then you would probably like a way to appease a user who says: “I like classic rock that uses lots of keyboard and slow rythm”!

The previous markdown cell gave my quantitative approach to this problem.  To further that discussion,another idea could be to introduce weights to keywords, depending on how discriptive the keyword is.  For example, the keyword 'pop' could be a little more descriptive than 'piano', since one defines a genre with the latter is simply one of the instruments involved.  In the above spotify user statement,'classic rock' is the main genre which the user is looking for, then 'keyboard' and 'slow rythm' are elements which belong in any genre, but of course could be used to narrow down once the main genre is decided by the algorithm.