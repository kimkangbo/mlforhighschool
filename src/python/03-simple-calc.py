import time  # To check how long this program to spend time

x_data = [100, 110, 120, 130, 140]    # Temperature Data
y_data = [12, 13.5, 13.6, 15.2, 15.7] # Measured resistance data on each teperature

print ("Temperature:", x_data) # Print x_data on screen
print ("Resistance:", y_data)    # Print y_data on screen

# Based on y = W*x + b
W_min = -10.0    # Set W Minimum value
W_max = 10.0     # Set W Maximum value
W_delta = 0.01 # Set W incremental value
W_list = []     # Empty list for W values

b_min = -10.0    # Set b Minimum value
b_max = 10.0     # Set b Maximum value
b_delta = 0.01 # Set b incremental value
b_list = []     # Empty list for b values


# Make W list with incremental value
W_value = W_min
while W_value <= W_max:
    W_list.append(W_value)
    W_value += W_delta

# Make b list with incremental value
b_value = b_min
while b_value <= b_max:
    b_list.append(b_value)
    b_value += b_delta

SE_final = 0.0                               # Initialize SE_final (Squared Error) to zero
W_final = 0.0
b_final = 0.0

start_time = time.time()                    # Check start time for calculating
total_calc_count = 0

for W_index in range(len(W_list)):         # Run loop for W
    W = W_list[W_index]
    
    for b_index in range(len(b_list)):     # Run loop for b
        b = b_list[b_index]

        SE = 0.0                            # Initialize SE (Squared Error) to zero        
        for data_index in range(len(x_data)): # Run loop for x_data
            x = x_data[data_index]            
            y = W*x + b                       # Calculate linear equation with W, x, b
            err = y_data[data_index] - y      # Calculate error equation
            SE += pow(err, 2)                 # Sum of err^2 for x data
            total_calc_count += 1             # Add +1 to tocal_calc_count
            
        if (W_index == 0 and b_index==0) or (SE_final > SE): # Set 1st SE to SE_old and SE_old > SE
            SE_final = SE                     # Set new SE_final to SE
            W_final = W                       # Set new W_final to W
            b_final = b                       # Set new b_final to b
            
end_time = time.time()                      # Check end time for calculating
MSE = SE_final/len(x_data)                  # Calculate MSE (Mean Squared Error)
RMSE = pow(MSE, 0.5)                        # Calculate RMSE (Root Mean Squared Error)

print ("It took %d seconds." % (end_time-start_time))
print ("It took %s loops to calculate." % (format(total_calc_count, ",")))
print ("W: %.2f, b:%.2f, SE:%f, MSE: %f, RMSE: %f" % (W_final, b_final, SE_final, MSE, RMSE))
