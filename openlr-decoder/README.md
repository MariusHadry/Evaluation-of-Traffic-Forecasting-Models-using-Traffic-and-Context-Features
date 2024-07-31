# openlr-decoder

This project is forked from [here](https://github.com/FraunhoferIVI/openlr/). Some changes were made so that the project complies with the own structure of the database.
The initial project displays live traffic disruptions on a map. The traffic disruptions are retrieved from the HERE Traffic API and are displayed on a map by decoding the
corresponding olr codes. The code is modified to simply decode OLR-codes and save the mapping between the OLR-codes and the respective found edges in the graph
in an additional table.

