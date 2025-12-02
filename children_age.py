"""
Children Height vs Age Visualization
This program depicts children's height based on their age (0-18 years)
using WHO growth standards data for boys and girls.
"""

import matplotlib.pyplot as plt
import numpy as np

# Age data (in years)
ages = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

# Average height data for boys (in cm) - based on WHO growth standards
boys_height = np.array([
    50,    # Birth
    75,    # 1 year
    87,    # 2 years
    96,    # 3 years
    103,   # 4 years
    110,   # 5 years
    116,   # 6 years
    122,   # 7 years
    128,   # 8 years
    133,   # 9 years
    138,   # 10 years
    143,   # 11 years
    149,   # 12 years
    156,   # 13 years
    164,   # 14 years
    170,   # 15 years
    173,   # 16 years
    175,   # 17 years
    176    # 18 years
])

# Average height data for girls (in cm) - based on WHO growth standards
girls_height = np.array([
    49,    # Birth
    74,    # 1 year
    86,    # 2 years
    95,    # 3 years
    102,   # 4 years
    109,   # 5 years
    115,   # 6 years
    121,   # 7 years
    127,   # 8 years
    133,   # 9 years
    138,   # 10 years
    144,   # 11 years
    151,   # 12 years
    157,   # 13 years
    160,   # 14 years
    162,   # 15 years
    163,   # 16 years
    163,   # 17 years
    163    # 18 years
])

def plot_height_vs_age():
    """
    Create a visualization showing the relationship between 
    children's height and age for both boys and girls.
    """
    # Create figure with high DPI for better quality
    plt.figure(figsize=(12, 7))
    
    # Plot boys' data
    plt.plot(ages, boys_height, 'b-o', linewidth=2, markersize=6, 
             label='Boys', alpha=0.8)
    
    # Plot girls' data
    plt.plot(ages, girls_height, 'r-s', linewidth=2, markersize=6, 
             label='Girls', alpha=0.8)
    
    # Customize the plot
    plt.xlabel('Age (years)', fontsize=12, fontweight='bold')
    plt.ylabel('Height (cm)', fontsize=12, fontweight='bold')
    plt.title('Children Height Based on Age\n(WHO Growth Standards)', 
              fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    plt.legend(fontsize=11, loc='upper left')
    
    # Set axis limits with some padding
    plt.xlim(-0.5, 18.5)
    plt.ylim(40, 185)
    
    # Add horizontal lines for reference heights
    reference_heights = [50, 100, 150]
    for height in reference_heights:
        plt.axhline(y=height, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    
    plt.tight_layout()
    plt.show()

def get_height_prediction(age, gender='boy'):
    """
    Predict height for a given age and gender using linear interpolation.
    
    Parameters:
    -----------
    age : float
        Age in years (0-18)
    gender : str
        'boy' or 'girl'
    
    Returns:
    --------
    float : Predicted height in cm
    """
    if age < 0 or age > 18:
        return "Age must be between 0 and 18 years"
    
    if gender.lower() == 'boy':
        height = np.interp(age, ages, boys_height)
    elif gender.lower() == 'girl':
        height = np.interp(age, ages, girls_height)
    else:
        return "Gender must be 'boy' or 'girl'"
    
    return round(height, 1)

def print_height_table():
    """
    Print a formatted table showing height data for all ages.
    """
    print("\n" + "="*50)
    print("CHILDREN HEIGHT BASED ON AGE (WHO Standards)")
    print("="*50)
    print(f"{'Age (years)':<15} {'Boys (cm)':<15} {'Girls (cm)':<15}")
    print("-"*50)
    
    for i in range(len(ages)):
        print(f"{ages[i]:<15} {boys_height[i]:<15} {girls_height[i]:<15}")
    
    print("="*50 + "\n")

def interactive_height_check():
    """
    Interactive function to check expected height for a specific age and gender.
    """
    print("\n--- Height Predictor ---")
    try:
        age = float(input("Enter child's age (0-18 years): "))
        gender = input("Enter gender (boy/girl): ").strip().lower()
        
        height = get_height_prediction(age, gender)
        
        if isinstance(height, str):
            print(height)
        else:
            print(f"\nExpected height for a {age}-year-old {gender}: {height} cm")
            print(f"That's approximately {height/2.54:.1f} inches")
    except ValueError:
        print("Invalid input. Please enter a valid number for age.")

if __name__ == "__main__":
    # Print the height table
    print_height_table()
    
    # Create the visualization
    plot_height_vs_age()
    
    # Optional: Run interactive height check
    # Uncomment the lines below to enable interactive mode
    # while True:
    #     interactive_height_check()
    #     continue_check = input("\nCheck another height? (yes/no): ").strip().lower()
    #     if continue_check != 'yes':
    #         break
    
    print("\nProgram completed successfully!")

