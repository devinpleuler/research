## Fixed to Fluid: Frame-by-Frame Role Classification

##### **Devin Pleuler | June 2024**

Soccer is often compared to chess. Pregame lineups on television broadcasts resemble the start of a chess game, with rigid and symmetrical lines of actors categorized into different functions and affixed to either side of the playing surface. Unfortunately, the parallels quickly melt away after kickoff.

The rigours of modern soccer have required players to become increasingly flexible, often occupying multiple roles as the game unfolds. This requirement has been exacerbated by the growing trend of teams playing dynamic and asymmetrical formations, with team technicians building rotations directly into their game models.

For example, a player assigned as a right defender may be asked to present themselves as a right winger when the true winger pinches inward and behaves like an attacking central midfielder. This is one of the most common rotations in modern soccer today, but there are plenty of others.

![](https://github.com/devinpleuler/research/blob/master/src/4231.png)

Technical staff monitor these rotations but they're difficult to systematically catalogue amidst the chaotic environment of a soccer game, especially as they can cause cascading role reassignment across an entire team. It would be useful to automatically detect and analyze these positional rotations for a multitude of reasons.

There is already some outstanding research on team structure. My particular favourite being Dynamic Analysis of Team Strategy in Professional Football by Shaw & Glickman. I've borrowed bits and pieces from that paper. In particular, the observation of team formation as a set of player-wise bivariate distributions. While Shaw and Glickman approach this in a more complicated manner, I take an easier path by normalizing player positions relative to a team centroid (like Gregory does in Ready Player Run: Off-ball run identification and classification). This dramatically minimizes the covariance of individual player distributions.

![](https://github.com/devinpleuler/research/blob/master/src/formations.png)

As opposed to traditional approaches, this style of visualization is helpful for demonstrating the variability that serves as the motivation for this work. But it's not perfect. The prevalence of rotations in modern tactical canon suggests that many player positions are not normally distributed, or entirely non-parametric. Depends on the player.

![](https://github.com/devinpleuler/research/blob/master/src/lw.png)

This underscores the utility in classifying the role that a particular player is occupying on a frame-by-frame basis.

To start, let's plant ourselves directly in the boots of a right defender.

![](https://github.com/devinpleuler/research/blob/master/src/rb.png)

Aside from the central defenders, most of your teammates are further upfield than you. And depending on if you're the "strong" or "weak" side defender, you might be further upfield than your opposite wide defender. Given that you're assigned to the right side, all of your teammates are to your left (while oriented upfield) except from the occasional winger deployed along your flank.

All of the information you need to determine your current tactical responsibility can be inferred from your relative spatial positioning from your teammates.

To encode this spatial awareness into machine-interpretable features, let's imagine our player rotating in place and gradually counting the number of teammates they can see in front of them.

![](https://github.com/devinpleuler/research/blob/master/src/rotate.gif)

Our right defender is going to have most of their teammates in view when they're facing the middle of the field but they won't see any teammates when they're oriented backwards toward their local sideline. 

Conversely, a more central player will probably have a few teammates in view no matter which direction they're facing but will rarely see all of them at once!

This relationship between angular orientation and visible teammate count is intuitively represented with radar charts. 

![](https://github.com/devinpleuler/research/blob/master/src/radar.png)

When unfurled, these waveforms have very soccer-interpretable properties. For example, a left and right defender would have similar amplitude, but they would be phase-shifted by half a period.

Features extracted from these waves can be used to assemble a classification model that can assign a current position to each player on a frame-by-frame basis.

For a supervised classification approach, we require ground truth labels to be aligned with the frame-wise player tracking data. Fortunately, most players occupy their assigned roles for a majority of the game. Positional rotations are increasingly common, but they are mostly fleeting and quickly revert into the status quo after a specific tactical moment has resolved.

For these labels, we utilize the position categorization conventions used in Statsbomb's event data specification as they are well balanced between generality and specificity. But of course, these methods will work just fine with alternative position maps.

The model selection to map the player-level spatial features to their positional labels is not particularly important. For the purposes of this research, we used a vanilla XGBoost classifier, but you can almost certainly achieve similar-to-better results with different flavours and sophistication of approaches.

![](https://github.com/devinpleuler/research/blob/master/src/xgb.png)

Here is an example inference, and the related tracking frame.

![](https://github.com/devinpleuler/research/blob/master/src/predictions.png)

Pretty nice! The model incorrectly assigns three labels, but they're all pretty sensical! It's also worth mentioning that this example is not in the training data set.

- The right defensive midfielder has dropped between the center backs, and the model thinks they're a central defender.
- The central attacking midfielder is sitting a little deeper on the left side, and is labelled as a left defensive midfielder.
- The left defensive midfielder has found themselves on their opposite side, and the model assumes they're a right defensive midfielder.

We can construct a confusion matrix after running these inferences for every frame across an entire game to understand how well the model is performing.

Notice that the matrix is truncated. This is because not every position label has a corresponding player sample in this particular game (and some true position labels actually don't exist in the training data). In this example, the team did not line up with a central defender, but players were occasionally classified as such over the course of the game.

![](https://github.com/devinpleuler/research/blob/master/src/confusion.png)

This is quite helpful for validating the behaviour of our model. The left defensive midfielder very rarely presents themselves as a center back. Conversely, the right defensive midfielder quite frequently is found in a position that resembles a center back. This sort of asymmetry can provide vital insight into a team's preferred tactical rotations. 

The model does not perfectly predict goalkeeper labels, which is suggestive that we're not over-fitting. Since the input features are purely geometric, you can easily imagine the model being confused by the rare set piece situation where the goalkeeper may not be the closest player to their goal.

This sort of analysis can easily be conducted across an entire season of data. If aggregated correctly, it can be quite useful in various team operations contexts ranging from opposition analysis to player recruitment.

It can also be used in a bunch of other less obvious ways:

Error Correction

> Data collectors are the real heroes of the soccer analytics movement, but they're not perfect. Especially when it comes to categorizing team formations. Games that have higher-than-usual distance between manual-labels and predicted-labels might have an error worth correcting.

Garbage Time Indicator

> If you measure classification error over the course of a game, it explodes during set pieces and other moments where the game is particularly disorganized. It can be useful to exclude metrics accumulated during these moments from certain performance measures to prevent bias.

Detect Formation Changes

> I haven't tried this yet, but I'm pretty certain it would work. I'm not sure if it would be better than the Shaw & Glickman approach, but it should be pretty serviceable.

Embeddings for other models

> As demonstrated above, the model provides probabilities for each player and position ID pairing. This can be quite a useful feature vector for other models, and could be treated as an embedding.

Better Normalize Player Position Plots

> If you filter the qualifying points that produce a player's average position by the moments where they're classified into a certain role, you are going to get even more stable and tactically representative plots. The player distributions will behave more parametrically and the covariance shrinks.

While this example has demonstrated the concept with full team tracking, it can easily be extended to partial or inferred tracking data collected from broadcast footage with a few helper features such as the camera field of view.

With all of this practical utility of frame-level position modelling, it feels inevitable that this sort of approach will become a bit more common in the future. In fact, I would be thrilled if a data collector (i.e. Statsbomb) simply delivered predictions of this fashion alongside other foundational models (e.g. xG) in their existing data feeds.

---

This work was most recently presented at the Carnegie Mellon Sports Analytics conference in 2023, but I actually demonstrated a similar model at the Fields Sports Analytics Workshop all the way back in 2018. I'm glad to finally have the idea in a more publicly accessible form.
