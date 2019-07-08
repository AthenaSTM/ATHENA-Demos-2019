# ATHENA RESTful API Demo: Classifiers

The classifiers implemented in this demo are based on the papers:

+ Turner, Chakrabarti, Jones, Xu, Fox, Luger, Laird, and Turner (2013). [Automated annotation of functional imaging experiments via multi-label classification](https://www.frontiersin.org/articles/10.3389/fnins.2013.00240/full), Frontiers in Neuroscience.
+ Riedel, Salo, Hays, Turner, Sutherland, Turner, and Laird (2019). [Automated, efficient, and accelerated knowledge modeling of the cognitive neuroimaging literature using the ATHENA toolkit](https://www.frontiersin.org/articles/10.3389/fnins.2019.00494/abstract), Frontiers in Neuroscience.

Specifically, these classifiers were built using the data sets from the Riedel <i>et al.</i> paper. Classifiers are implemented for the [CogPO](http://www.cogpo.org) labels for which sufficient data were available (see the Riedel <i>et al.</i> supplemental materials online for more details). Only the naïve Bayes classifier types were implemented as these appear to be the most generically useful across both of the previous studies.

All of the current implementation is done in the Python language using standard libraries including Numpy, Scikit-Learn, and Pandas. Two sets of classifiers are available on this server. One is based on a full-text analysis of the papers used as training data, while the other is based on the text of abstracts of the papers. The implementations here were developed by the team at GSU, using the **full** data set, without holding out any data for testing. (Therefore the performance and results of classification will not be identical with the paper's results.)

_This research was funded by a grant from the National Science Foundation (NSF; USA) to Drs. J. Turner and A. Laird, grant number 1631325._

###### Matthew D. Turner<br>Georgia State University <br> 2019.07.03
