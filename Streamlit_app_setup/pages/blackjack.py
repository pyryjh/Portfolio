import streamlit as st
import pandas as pd
import numpy as np
import random

# Define the functions
def define_deck():
    columns = ['hearts', 'spades', 'clubs', 'diamonds']
    rows = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    df = pd.DataFrame(5, index=rows, columns=columns)
    return df

def shuffle_deck(df):
    result_list = []

    for row in df.index:
        for col in df.columns:
            count = df.loc[row, col]
            for _ in range(count):
                result_list.append([row, col])
    
    result_array = np.array(result_list)

    deck_list = []
    
    while len(result_array) > 0:
        random_index = random.randint(0, len(result_array) - 1)
        selected_row = result_array[random_index]
        deck_list.append(selected_row)
        result_array = np.delete(result_array, random_index, axis=0)
    
    deck = np.array(deck_list)
    return deck

def deal_initial(deck):
    player_list = []
    player_list.append(deck[0])
    player_list.append(deck[2])
    player_hand = np.array(player_list)
    dealer_list = []
    dealer_list.append(deck[1])
    dealer_list.append(deck[3])
    dealer_hand = np.array(dealer_list)
    return player_hand, dealer_hand

# Streamlit app
st.title("Card Dealer App")

# Button to deal cards
if st.button("Deal"):
    # Define and shuffle the deck
    deck = shuffle_deck(define_deck())
    
    # Deal the initial hands
    player_hand, dealer_hand = deal_initial(deck)
    
    # Display player and dealer hands
    st.write("Player's Hand:")
    st.write(player_hand)
    
    st.write("Dealer's Hand:")
    st.write(dealer_hand)