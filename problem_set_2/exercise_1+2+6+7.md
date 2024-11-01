# Exercise 1

## How can a competitor - in principle - try to steal the valuable data for recommendation from this website?

A competitor might try to reverse-engineer the recommendation system’s utility matrix by creating many new accounts and interacting with only one or a few items per account. By observing the system’s responses, they could infer relationships between items and users.

## Does this work better when the web shop implemented a content-based or a collaborative fitering system?

This approach would be more effective against a content-based filtering system, as content-based recommendations are driven by item features and similarities, making them easier to infer. Additionally, content-based systems often provide explanations, which could reveal more about the item features being used.

## What data would the competitor be able to infer?

The competitor could potentially infer patterns in the recommended items, revealing some of the features or relationships within the recommendation system. This could allow them to approximate the underlying item similarities or user preferences that guide recommendations.

## Would this technique have an impact on the recommendation system, i.e., would this attack create a bias on the data?

No, this approach would not directly impact the recommendation system’s accuracy or introduce bias, as it relies on isolated interactions that don’t substantially influence the overall user data.

## Why is this attack probably not viable in any case?

This approach is impractical because the recommendation system likely draws on a large, complex set of features and items. Reverse-engineering these relationships would require an enormous number of fake accounts and interactions to obtain even a rough estimate of the system’s internal values.

# Exercise 2

## a)

- AB = 0.3989592359914347
- AC = 0.38508130618755787
- BC = 0.48612988022263837

## b)

- AB = 0.4836022205056778
- AC = 0.40371520600005606
- BC = 0.42264973081037427

Now they are more dissimilar especially AB.

## c)

- AB = 0.41569345253185686
- AC = 1.1154700538379252
- BC = 1.7395739969534467

## d)

- AB = -0.04637388957601683
- AC = -0.037662178857735464
- BC = -0.28817938438423535

I think my Pearson correlation values are wrong, as these shor that there is no strong linear relationship between any of the pairs.

# Exercise 6

## a)

What data can you use to populate a utility matrix in the following cases?

- Numerical ratings for teaching style, clarity, difficulty etc.
- Qualitative rating by analyzing a written feedback and giving it a positive or negative rating based in the sentiment

What relations other than like or a numerical rating can be exploited to express a positive or negative connection?

- Students could recommend professors

## b)

What data can you use to populate a utility matrix in the following cases?

- Numerical rating (0/1) if the user dislikes or likes the artwork
- Number of views, comments and shares an artwork recieves

What relations other than like or a numerical rating can be exploited to express a positive or negative connection?

- If a user follows an artist or if he adds the artwork to his favorites

## c)

What data can you use to populate a utility matrix in the following cases?

- How many profile visits, likes and messages a profile gets
- How often a user is blocked

What relations other than like or a numerical rating can be exploited to express a positive or negative connection?

- The dream partner profile can be used to assess potential matches based on interests etc.
- Mutual interaction

What is special about this scenario?

- Special is that each user is also an item and vise versa

# Exercise 7

## a)

What would the lists of movies recommended by each system look like?

- G: Highly popular movies
- R: Mix of popular and niche movies based on user preferences
- L: Mix of popular and niche movies based on similar users preferences

Does any of the systems G, R and L introduce users to more exotic items from the long tail?

- The L system indroduces users to more exotic items

## b)

What system would be most robust to such an attempt?

- The most robust system would be the L one, as it relies on user behavior patterns.

Which users recommendation lists would be affected most?

- The user who prefers the mainstream films (G)

## c)

Can these types of users benet from either of the three systems?

- gray sheep benefit from the R and L system
- black sheep benefit from the R and L system too
