def calculate_outlier_percentage():
            if len(data) < 4:
                return 0.0
            data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300]
            sorted_data = sorted(data)
            n = len(sorted_data)
            k = 1.5
            
            # Calculate Q1 and Q3
            q1 = sorted_data[n//4]
            q3 = sorted_data[3*n//4]
            iqr = q3 - q1
            
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            
            outlier_count = sum(1 for x in data if x < lower_bound or x > upper_bound)
            percentage = (outlier_count / len(data)) * 100
            
            print(f"Problem 5 - Input: {data}")
            print(f"Problem 5 - Outlier percentage: {percentage}%")
    
    def cacluate_median():
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median = sorted_data[n//2]
        print(f"Problem 5 - Input: {data}")
        print(f"Problem 5 - Median: {median}")
    
    def calculate_mean():
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300]
        mean = sum(data) / len(data)
        print(f"Problem 5 - Input: {data}")
        print(f"Problem 5 - Mean: {mean}")
    
    def calculate_variance():
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        print(f"Problem 5 - Input: {data}")
        print(f"Problem 5 - Variance: {variance}")
    
    def calculate_standard_deviation():
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        standard_deviation = variance ** 0.5
        print(f"Problem 5 - Input: {data}")
        print(f"Problem 5 - Standard Deviation: {standard_deviation}")
    
    def calculate_z_score():
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        standard_deviation = variance ** 0.5
        z_score = (data - mean) / standard_deviation
        print(f"Problem 5 - Input: {data}")
        print(f"Problem 5 - Z-Score: {z_score}")
        
    def division_methods():
        number = 5
        divisor = 2
        quotient = number // divisor
        remainder = number % divisor
        print(f"Problem 5 - Input: {number}")
        print(f"Problem 5 - Quotient: {quotient}")
        print(f"Problem 5 - Remainder: {remainder}")
        
    def power_methods():    
        number = 2
        power = 3
        result = number ** power
        print(f"Problem 5 - Input: {number}")
        print(f"Problem 5 - Power: {power}")
        print(f"Problem 5 - Result: {result}")
        
    def root_methods():
        number = 27
        root = 3
        result = number ** (1/root)
        print(f"Problem 5 - Input: {number}")
        print(f"Problem 5 - Root: {root}")
        print(f"Problem 5 - Result: {result}")
        
    def calculate_q1():
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sorted_data = sorted(data)
        n = len(sorted_data)
        q1 = sorted_data[n//4]
        print(f"Problem 5 - Input: {data}")
        print(f"Problem 5 - Q1: {q1}")
        
    def calculate_q3():
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sorted_data = sorted(data)
        n = len(sorted_data)
        q3 = sorted_data[3*n//4]
        print(f"Problem 5 - Input: {data}")
        print(f"Problem 5 - Q3: {q3}")
        
    def calculate_iqr():
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sorted_data = sorted(data)
        n = len(sorted_data)
        q1 = sorted_data[n//4]
        q3 = sorted_data[3*n//4]
        iqr = q3 - q1
        print(f"Problem 5 - Input: {data}")
        print(f"Problem 5 - IQR: {iqr}")
        
        data_array = np.array(data)
        q1 = np.percentile(data_array, 25)
        q2 = np.percentile(data_array, 50)  # Median
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        print(f"Problem 5 - Input: {data}")
        print(f"Problem 5 - IQR: {iqr}")
        
    def calculate_z_score():
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        standard_deviation = variance ** 0.5
        z_score = (data - mean) / standard_deviation
        print(f"Problem 5 - Input: {data}")
        print(f"Problem 5 - Z-Score: {z_score}")
    
    
    
    if __name__ == "__main__":
        calculate_outlier_percentage() # using iqr method to calculate outlier percentage