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