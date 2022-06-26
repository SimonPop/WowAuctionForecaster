# 📈 World of Warcraft Auction House Forecasting

## Introduction

This project aims at providing a forecasting model for the World of Warcraft Auctions house.

The model is based on a Temporal Graph Neural Network.

>__Why Temporal? :__ Because we want to forecast prices at the auction house, using time series of the previous prices.

> __Why Graph? :__ Because we want to use the explicit links that exist between different product in the game. (e.g. An object in World of Warcraft may be created from a list of components, creating a direct relation between prices of these components and the object itself).    

## 📂 Data 

Two sources of data are used for this project.

### 1. Component <-> Product data

This source of data is used in order to create the graph that links products and components to each other. This will be used as the backbone for the GNN.

>The data comes from the famous [Wowhead](www.wowhead.com) wiki website.

### 2. Time series data

This source of data is used for the prices of all products in the auction house.

> The data comes from [The Undermine Journal](https://theunderminejournal.com/) and its SQL database.


## ⚡️ Model

The model used was inspired by an early bike usage prediction model for London shared bikes.

This model was build using the [PyG library](https://pytorch-geometric.readthedocs.io/en/latest/).
