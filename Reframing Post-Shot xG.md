## **Reframing Post-Shot xG**

#### ***Devin Pleuler | March 2022***

Expected Goals are ubiquitous. It's jumped off the esoteric analytics blog posts of the 2010's and on to your television screens and into your video games, but also sporting club databases and pitch-side tablets. 

Its rapid proliferation can perhaps be attributed to the same reason why its provenance is so hard to nail down: it's painfully intuitive. Estimating the danger of every shot or sequence is the underlying job of every player, coach, and analyst alike. It's shocking that we didn't think of it earlier.

While the utility and precision of individual xG measurements still depend on the quality of the features that inform it, most analysts would claim that the true innovation of xG is the action-value framework that records the fluctuations of goalscoring probability as a possession unfolds.

Looking backward before shots are taken: xG is used as a unit in Expected Possession Value (i.e. EPV, or others like xT or VAEP) which is used to estimate the value gained by moving the ball from low-danger to high-danger situations. This is quite useful for appreciating the value added by players who aren't typically directly involved in the goalscoring process, like a talented defensive midfielder who breaks line after line. 

Looking forward, after a shot is taken: the framework has been tweaked to update the original xG measure to incorporate the post-shot ball trajectory. For example, an 0.10 xG shot might become a 0.50 xG shot if it's stuck toward the upper corner of the goal. By comparing the predicted measures of xG and PSxG (i.e. Post-Shot Expected Goals) you can get a grasp of which shots are well-taken, and by extension: the cumulative value added by players who have taken them.

(And naturally, the next step in this value framework leads us to `G-PSxG`, which is quite useful for evaluating goalkeepers)

![](https://github.com/devinpleuler/research/blob/master/src/xg_value_progression.png)

It's an elegant framework for ~~a more civilized age~~ action valuation that easily incorporates new innovations and increasingly sophisticated models as they become available. As tracking data becomes increasingly common in recruitment settings, off-ball player positioning will strengthen the underlying models and the framework will remain sturdy.

But, there is a problem with the sharp end of the framework. The value that a player adds via shooting does not appear to be stable on a year-to-year basis. A positive `(PSxG - xG) / Shot` in one year does not seem to increase the chances of a positive residual in the following year. This leads to the troublesome implication which suggests finishing ability isn't as much a repeatable skill as it is a random effect. This heresy would get any analyst and their TI-84+ tossed out of any scouting meeting.

The purpose of this article is to suggest that our dogmatic reliance on the traditional Expected Goals framework has led us in the wrong direction when trying to disentangle finishing skill from the rest of the goalscoring process.

The probability of an off-target shot resulting in a goal (sans deflection) is zero. Therefore, Post-Shot Expected Goals models traditionally assign a zero to all shots that are off-frame in an effort to minimize error.

While this modelling decision is reasonable for the primary purpose of predicting goals on a shot-by-shot basis, it has dramatic repercussions for predicting future goals. The problem is that all off-target shots are treated (i.e. penalized) the same, no matter if they graze the outside of the post or if they become the latest satellite in low-Earth orbit. This makes sense if your aim is to predict goal totals from already-taken shots, but it doesn't make sense for predicting future performance. A shot that narrowly misses the goal should garner more confidence in future success than a shot that misses by a wide margin. These are not-so-subtle hints that we're overfitting in some sense.

This issue is exacerbated by a lack of sample size. From a `goals / shot` perspective,  players rarely attempt enough shots in a season to reliably conclude that they are above-or-below average in terms of finishing with any sort of statistical significance. If we had huge samples, this problem likely goes away.

To demonstrate this effect, here is a cherry-picked visual:
![](https://github.com/devinpleuler/research/blob/master/src/example.png)
Randomly sampling with `n=10`, the player with the worse accuracy (`σ = 0.7 meters`) can easily have a higher Post-Shot xG value despite it being pretty clear that the other player with the better underlying accuracy (`σ = 0.5 meters`) is likely to score more goals across a larger sample size.

The best way to combat this overfitting is the allow off-target shots to retain some xG value; in turn crediting narrow misses with more value than wide misses. In other words, we need to extract more meaning out of every sample. However, it's not completely intuitive how this makes any conceptual sense.

To demonstrate, we train a goal expectation model that estimates xG based on the shot trajectory error, as opposed to just using the shot destination. This makes sense if we're interested in rewarding process instead of results. Ultimately we want to credit a player who strikes the ball accurately, no matter which side of the post it happens to end up.

To test this, we fit a vanilla XGBoost model with two variables: Expected Goals (`xg`) and the distance between the shot destination and 0.5 meters inside the nearest frame of the goal (`r`) and try to predict Goals.

```python
model = xgboost.XGBClassifier(max_depth=2)
X = df[['xg', 'r']]
y = df['goal']
model.fit(X, y)
```

At its core, this model is very simple. It pairs an initial xG measure with a shot trajectory error and spits out a different flavor of Post-Shot xG. However, we can exploit the spatial nature of the trajectory errors and evaluate the model across all z-coordinates along the goal line axis to create a value surface. Adding a contour plot for aesthetics, below is what the value distribution looks for a shot with an initial `xG=0.4` for actual shot destinations. It's not perfect, but sensical!
![](https://github.com/devinpleuler/research/blob/master/src/xg04.png)
And this is what it looks like with `xG=0.1`:
![](https://github.com/devinpleuler/research/blob/master/src/xg01.png)As this is a toy model for demonstration purposes, we've thrown an easy-to-implement classification model at this. XGBoost can easily be replaced with alternatives as simple as a logistic regression, or something wildly more sophisticated. But regardless of model architecture, the trick here is in preventing the model from learning too much by engineering your features carefully.

Now, it's worth comparing this "naive" Post-Shot xG model to both classic xG models and Post-Shot xG Models. Below are the ROC curves for these three models.

![](https://github.com/devinpleuler/research/blob/master/src/roc.png)
As you would expect, when used to predict goals, our Naive PSxG model performs better than raw Expected Goals (as it includes some trajectory information), but not as strongly as the standard Post-Shot Expected Goals (as it includes less trajectory information).

> As a note: I recognize that I'm not using "naive" in the fashion it is typically used in statistical jargon. It's more of a literal description as I am intentionally withholding some of the spatial context from its input features. Would love to hear different naming ideas.

But does it do a better job at predicting future performance? Well, this is a more difficult question than it seems and probably needs someone to do a more rigorous (and frankly, more academic) study than mine.

However, I am confident that `(Naive PSxG - xG) / shot` is considerably more stable on a player-level basis than  `(PSxG - xG) / shot`, at least across my sample of 269 players who have attempted at least 50 shots.

![](https://github.com/devinpleuler/research/blob/master/src/stability.png)

To briefly explain the methodology: for every qualifying player, I've taken their n shots, shuffled them, and split them into two equally sized samples. In the visual, I've plotted the mean residuals in each sample across each other.

If it isn't already clear, the naive approach provides considerably more stability – where the traditional approach, as alluded to above, has practically no repeatability (`r_values = 0.03, 0.57`). For certain use cases, this Naive approach to Post-Shot Expected Goals provides the best of both worlds: the model precision of PSxG without sacrificing the player-level stability of xG.

There remains a lot of room for improvement here, but I think this is compelling evidence that this sort of modelling approach may prove fruitful for better understanding underlying finishing ability.