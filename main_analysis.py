import pandas as pd
import re
from datetime import datetime
import json
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple
from scipy import stats


class DifficultyAnalyzer:
    def __init__(self):
        self.patterns = {
            # General patterns for expression counting
            'arithmetic': r'\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?',
            'variable_assignment': r'\w+\s*=\s*[\d\w\+\-\*\/\(\)\s\.]+',
            'percentage': r'\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*percent',
            'fractions': r'\d+\/\d+',
            'decimal_mult': r'\d+(?:\.\d+)?\s*\*\s*0\.\d+',
            'complex_arithmetic': r'\([\d\s\+\-\*\/\.]+\)',
            # Specific operation patterns for difficulty calculation
            'addition': r'\d+(?:\.\d+)?\s*\+\s*\d+(?:\.\d+)?',
            'subtraction': r'\d+(?:\.\d+)?\s*\-\s*\d+(?:\.\d+)?',
            'multiplication': r'\d+(?:\.\d+)?\s*\*\s*\d+(?:\.\d+)?',
            'division': r'\d+(?:\.\d+)?\s*\/\s*\d+(?:\.\d+)?',
        }

    def normalize_expression(self, expr: str) -> str:
        """Normalize expression to avoid duplicates"""
        # Remove spaces
        expr = re.sub(r'\s+', '', expr)
        # Remove parentheses for comparison
        expr = re.sub(r'[()]', '', expr)
        return expr

    def count_expressions(self, solution: str) -> Tuple[int, List[str]]:
        if not isinstance(solution, str):
            return 0, []

        expressions = []
        seen_normalized = set()

        # Process solution line by line
        for line in solution.split('\n'):
            # Skip empty lines and lines that are just comments
            if not line.strip() or line.strip().startswith('#'):
                continue

            # Find all expressions in this line
            for pattern_name, pattern in self.patterns.items():
                found = re.findall(pattern, line)
                for expr in found:
                    expr = expr.strip()
                    if expr:
                        # Only add if normalized version hasn't been seen
                        normalized = self.normalize_expression(expr)
                        if normalized not in seen_normalized:
                            expressions.append(expr)
                            seen_normalized.add(normalized)

        return len(expressions), expressions

    def detect_operation_types(self, solution: str) -> dict:
        """
        Detect operation types present in the solution.

        Args:
            solution (str): The solution text to analyze

        Returns:
            dict: Boolean values indicating which operations are present
        """
        if not isinstance(solution, str):
            return {
                'has_addition': False,
                'has_subtraction': False,
                'has_multiplication': False,
                'has_division': False
            }

        operations = {
            'has_addition': bool(re.search(self.patterns['addition'], solution)),
            'has_subtraction': bool(re.search(self.patterns['subtraction'], solution)),
            'has_multiplication': bool(re.search(self.patterns['multiplication'], solution)),
            'has_division': bool(re.search(self.patterns['division'], solution))
        }

        return operations

    def calculate_difficulty(self, solution: str) -> int:
        """
        Calculate difficulty level based on operation types:
        Level 1: Addition and/or subtraction only
        Level 2: Multiplication OR division (but not both)
        Level 3: Both multiplication AND division

        Args:
            solution (str): The solution text to analyze

        Returns:
            int: Difficulty level (1-3)
        """
        operations = self.detect_operation_types(solution)

        if operations['has_multiplication'] and operations['has_division']:
            return 3
        elif operations['has_multiplication'] or operations['has_division']:
            return 2
        elif operations['has_addition'] or operations['has_subtraction']:
            return 1
        else:
            # Default difficulty for solutions with no detected operations
            return 1

    def analyze_solution(self, solution: str) -> dict:
        """
        Analyze a solution and return difficulty metrics.
        """
        num_expr, expressions = self.count_expressions(solution)
        difficulty = self.calculate_difficulty(solution)
        operations = self.detect_operation_types(solution)

        return {
            'num_expressions': num_expr,
            'difficulty_level': difficulty,
            'expressions': expressions,
            'operations': operations
        }

    def batch_analyze(self, df, answer_column='answer'):
        """
        Analyze all solutions in a DataFrame using operation-based difficulty.
        """
        # Analyze each solution
        results = []
        for idx, row in df.iterrows():
            analysis = self.analyze_solution(row[answer_column])
            results.append(analysis)

        # Add results to DataFrame
        df['num_expressions'] = [r['num_expressions'] for r in results]
        df['difficulty_level'] = [r['difficulty_level'] for r in results]
        df['has_addition'] = [r['operations']['has_addition'] for r in results]
        df['has_subtraction'] = [r['operations']['has_subtraction'] for r in results]
        df['has_multiplication'] = [r['operations']['has_multiplication'] for r in results]
        df['has_division'] = [r['operations']['has_division'] for r in results]

        # Print distribution for verification
        print("\nDifficulty Level Distribution:")
        print(df['difficulty_level'].value_counts().sort_index())

        # Print operation type distribution
        print("\nOperation Type Distribution:")
        print(f"Addition: {df['has_addition'].sum()}")
        print(f"Subtraction: {df['has_subtraction'].sum()}")
        print(f"Multiplication: {df['has_multiplication'].sum()}")
        print(f"Division: {df['has_division'].sum()}")

        return df


class PatternAnalyzer:
    def __init__(self):
        # Patterns for different types of backtracking
        self.backtrack_types = {
            'numerical_correction': [
                r'actually (?:equals|is) (?:-?\d*\.?\d+)',
                r'should be (?:-?\d*\.?\d+)',
                r'I calculated wrong',
                r'incorrect calculation'
            ],
            'operation_correction': [
                r'should (multiply|divide|add|subtract)',
                r'wrong operation',
                r'instead of (multiplication|division|addition|subtraction)',
                r'used the wrong operator'
            ],
            'assumption_revision': [
                r'forgot to consider',
                r'didn\'t account for',
                r'missed a step',
                r'overlooked'
            ],
            'verification': [
                r'let me verify',
                r'double[- ]check',
                r'make sure',
                r'confirm'
            ]
        }

        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE)
                       for pattern in patterns]
            for category, patterns in self.backtrack_types.items()
        }

    def find_number_changes(self, steps):
        """
        Identify number changes before and after backtracking.

        Args:
            steps: List of step dictionaries from the BacktrackAnalyzer

        Returns:
            List of dictionaries containing information about number changes
        """
        changes = []

        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]

            # Only analyze if next step is a backtrack
            if next_step['is_backtrack']:
                current_numbers = set(float(n) for n in current_step['numbers'])
                next_numbers = set(float(n) for n in next_step['numbers'])

                # Find changed numbers
                changed_numbers = []
                for num in current_numbers:
                    # Look for similar but not identical numbers in next step
                    for next_num in next_numbers:
                        if num != next_num and abs(num - next_num) / max(abs(num), 1) < 0.1:
                            changed_numbers.append({
                                'old_value': num,
                                'new_value': next_num,
                                'position': next_step['position'],
                                'context': next_step['text']
                            })

                if changed_numbers:
                    changes.append({
                        'step_index': i,
                        'position': next_step['position'],
                        'changes': changed_numbers
                    })

        return changes

    def classify_backtrack(self, step_text):
        """
        Classify the type of backtracking based on the text.

        Args:
            step_text: Text of the step containing backtracking

        Returns:
            List of detected backtrack types
        """
        detected_types = []

        for category, patterns in self.compiled_patterns.items():
            if any(pattern.search(step_text) for pattern in patterns):
                detected_types.append(category)

        return detected_types if detected_types else ['unclassified']

    def analyze_patterns(self, solution_steps):
        """
        Analyze patterns in the solution steps.

        Args:
            solution_steps: List of step dictionaries from the BacktrackAnalyzer

        Returns:
            Dictionary containing pattern analysis results
        """
        # Initialize results structure
        analysis = {
            'number_changes': self.find_number_changes(solution_steps),
            'backtrack_types': [],
            'operation_sequences': [],
            'backtrack_positions': []
        }

        # Analyze each step
        for i, step in enumerate(solution_steps):
            if step['is_backtrack']:
                # Classify backtrack type
                backtrack_type = self.classify_backtrack(step['text'])
                analysis['backtrack_types'].append({
                    'position': step['position'],
                    'types': backtrack_type,
                    'text': step['text']
                })

                # Record position
                analysis['backtrack_positions'].append(step['position'])

                # Record operation sequence if available
                if step['operation'] and i > 0 and solution_steps[i - 1]['operation']:
                    analysis['operation_sequences'].append({
                        'before': solution_steps[i - 1]['operation'],
                        'after': step['operation'],
                        'position': step['position']
                    })

        return analysis

    def generate_pattern_stats(self, all_patterns):
        """
        Generate statistics about detected patterns.

        Args:
            all_patterns: List of pattern analysis results from multiple solutions

        Returns:
            Dictionary containing pattern statistics
        """
        stats = {
            'backtrack_types_count': {},
            'operation_changes': {},
            'position_distribution': {
                'early': 0,  # 0-0.33
                'middle': 0,  # 0.33-0.66
                'late': 0  # 0.66-1.0
            },
            'number_changes_stats': {
                'total_changes': 0,
                'average_magnitude': 0.0
            }
        }

        total_magnitude = 0
        magnitude_count = 0

        for pattern in all_patterns:
            # Count backtrack types
            for bt in pattern['backtrack_types']:
                for t in bt['types']:
                    stats['backtrack_types_count'][t] = stats['backtrack_types_count'].get(t, 0) + 1

            # Count operation changes
            for op_seq in pattern['operation_sequences']:
                key = f"{op_seq['before']} → {op_seq['after']}"
                stats['operation_changes'][key] = stats['operation_changes'].get(key, 0) + 1

            # Count position distribution
            for pos in pattern['backtrack_positions']:
                if pos < 0.33:
                    stats['position_distribution']['early'] += 1
                elif pos < 0.66:
                    stats['position_distribution']['middle'] += 1
                else:
                    stats['position_distribution']['late'] += 1

            # Calculate number change statistics
            for change in pattern['number_changes']:
                for c in change['changes']:
                    stats['number_changes_stats']['total_changes'] += 1
                    magnitude = abs(c['new_value'] - c['old_value'])
                    total_magnitude += magnitude
                    magnitude_count += 1

        # Calculate average magnitude of changes
        if magnitude_count > 0:
            stats['number_changes_stats']['average_magnitude'] = total_magnitude / magnitude_count

        return stats

    def plot_pattern_distribution(self, pattern_stats, save_path=None):
        """
        Create visualization of pattern distribution.

        Args:
            pattern_stats: Statistics generated by generate_pattern_stats
            save_path: Optional path to save the plot
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot backtrack types distribution
        types_df = pd.DataFrame({
            'Type': list(pattern_stats['backtrack_types_count'].keys()),
            'Count': list(pattern_stats['backtrack_types_count'].values())
        })
        types_df = types_df.sort_values('Count', ascending=False)

        sns.barplot(data=types_df, x='Count', y='Type', ax=ax1)
        ax1.set_title('Distribution of Backtrack Types')

        # Plot position distribution
        pos_df = pd.DataFrame({
            'Position': list(pattern_stats['position_distribution'].keys()),
            'Count': list(pattern_stats['position_distribution'].values())
        })

        sns.barplot(data=pos_df, x='Position', y='Count', ax=ax2)
        ax2.set_title('Distribution of Backtrack Positions')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class BacktrackAnalyzer:
    def __init__(self):
        # Define patterns for backtracking moments
        self.backtrack_patterns = [
            r'wait',
            r'uhm',
            r'let me',
            r'hold on',
            r'let\'s see',
            r'hmm',
            r'checking',
            r'double[- ]check',
            r'verify'
        ]
        self.backtrack_regex = re.compile('|'.join(self.backtrack_patterns), re.IGNORECASE)

        # Define patterns for mathematical operations
        self.operations = {
            'multiplication': r'\*|times|multiply|product',
            'division': r'÷|divide|divided by|ratio',
            'addition': r'\+|plus|sum|add',
            'subtraction': r'-|minus|subtract|difference'
        }

    def plot_operations_distribution(self, results, save_path=None):
        """
        Plot the distribution of mathematical operations associated with backtracking.

        Args:
            results: List of analysis results
            save_path: Optional path to save the plot
        """
        # Collect all operations that occurred during backtracking
        operations = []
        for result in results:
            ops = [op for op in result['operations_with_backtrack'] if op is not None]
            operations.extend(ops)

        if not operations:
            print("No operations found in backtracking instances")
            return

        # Count occurrences of each operation
        op_counts = {}
        for op in set(operations):
            op_counts[op] = operations.count(op)

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Operation': list(op_counts.keys()),
            'Count': list(op_counts.values())
        })

        # Sort by count in descending order
        df = df.sort_values('Count', ascending=False)

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Operation', y='Count')
        plt.title('Mathematical Operations Associated with Backtracking')
        plt.xlabel('Operation Type')
        plt.ylabel('Number of Occurrences')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def analyze_solution(self, text):
        """Analyze a single solution text in detail"""
        if not isinstance(text, str):
            return []

        # Split solution into lines
        lines = text.split('\n')
        steps = []

        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue

            step_info = {
                'position': i / len(lines),  # Normalized position (0 to 1)
                'has_number': bool(re.search(r'\d+', line)),
                'is_backtrack': bool(self.backtrack_regex.search(line)),
                'numbers': re.findall(r'-?\d*\.?\d+', line),
                'text': line.strip(),
                'operation': self.detect_operation(line)
            }
            steps.append(step_info)

        return steps

    def detect_operation(self, line):
        """Detect what type of math operation is being performed in a line"""
        for op_name, pattern in self.operations.items():
            if re.search(pattern, line, re.IGNORECASE):
                return op_name
        return None

    def analyze_batch(self, df):
        """Analyze a batch of solutions from a DataFrame"""
        results = []

        for idx, row in df.iterrows():
            solution_steps = self.analyze_solution(row['full_response'])

            # Basic statistics for this solution
            backtrack_positions = [step['position'] for step in solution_steps if step['is_backtrack']]
            operations_with_backtrack = [step['operation'] for step in solution_steps if step['is_backtrack']]

            analysis = {
                'gsm8k_id': row['gsm8k_id'],
                'num_steps': len(solution_steps),
                'num_backtracks': len(backtrack_positions),
                'backtrack_positions': backtrack_positions,
                'operations_with_backtrack': operations_with_backtrack,
                'is_correct': row.get('is_correct', 0),
                'detailed_steps': solution_steps
            }

            results.append(analysis)

        return results

    def generate_stats(self, results):
        """Generate summary statistics from analysis results"""
        stats = {
            'total_solutions': len(results),
            'solutions_with_backtrack': sum(1 for r in results if r['num_backtracks'] > 0),
            'avg_backtracks_per_solution': sum(r['num_backtracks'] for r in results) / len(results),
            'backtrack_positions': {
                'early': sum(1 for r in results for pos in r['backtrack_positions'] if pos < 0.33),
                'middle': sum(1 for r in results for pos in r['backtrack_positions'] if 0.33 <= pos < 0.66),
                'late': sum(1 for r in results for pos in r['backtrack_positions'] if pos >= 0.66)
            },
            'accuracy': {
                'with_backtrack': sum(r['is_correct'] for r in results if r['num_backtracks'] > 0) /
                                  max(sum(1 for r in results if r['num_backtracks'] > 0), 1),
                'without_backtrack': sum(r['is_correct'] for r in results if r['num_backtracks'] == 0) /
                                     max(sum(1 for r in results if r['num_backtracks'] == 0), 1)
            },
            'operations_with_backtrack': {}
        }

        # Count operations that led to backtracking
        all_ops = [op for r in results for op in r['operations_with_backtrack'] if op is not None]
        for op in set(all_ops):
            stats['operations_with_backtrack'][op] = all_ops.count(op)

        return stats

    def plot_backtrack_distribution(self, results, save_path=None):
        """Plot the distribution of backtrack positions"""
        positions = [pos for r in results for pos in r['backtrack_positions']]

        plt.figure(figsize=(10, 6))
        plt.hist(positions, bins=20, edgecolor='black')
        plt.title('Distribution of Backtrack Positions in Solutions')
        plt.xlabel('Position in Solution (0=Start, 1=End)')
        plt.ylabel('Frequency')

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_accuracy_comparison(self, results, save_path=None):
        """Plot accuracy comparison between solutions with and without backtracking"""
        with_backtrack = [r['is_correct'] for r in results if r['num_backtracks'] > 0]
        without_backtrack = [r['is_correct'] for r in results if r['num_backtracks'] == 0]

        data = {
            'Accuracy': [sum(with_backtrack) / len(with_backtrack) if with_backtrack else 0,
                         sum(without_backtrack) / len(without_backtrack) if without_backtrack else 0],
            'Category': ['With Backtracking', 'Without Backtracking']
        }

        plt.figure(figsize=(8, 6))
        sns.barplot(x='Category', y='Accuracy', data=pd.DataFrame(data))
        plt.title('Solution Accuracy With vs Without Backtracking')
        plt.ylabel('Accuracy Rate')

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


class ResultAnalyzer:
    def __init__(self):
        # Setup base directory structure
        self.base_dir = Path.cwd()
        self.input_dir = self.base_dir / "input"
        self.backtrack_analyzer = BacktrackAnalyzer()

        # Create input directory if needed
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def create_difficulty_comparison_plot(self, df: pd.DataFrame, save_path=None):
        # Import scipy.stats for the t-test
        from scipy import stats
        """
        Create a visualization comparing success rates between backtracking and
        no backtracking across different difficulty levels, including t-test results.

        Args:
            df: DataFrame with columns 'is_correct', 'backtrack', and 'difficulty_level'
            save_path: Optional path to save the figure
        """
        # Ensure the needed columns exist
        required_cols = ['is_correct', 'backtrack', 'difficulty_level']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain all of these columns: {required_cols}")

        # Convert boolean backtrack to string for better labels
        df = df.copy()
        df['backtrack_label'] = df['backtrack'].map({True: 'Backtracking', False: 'No Backtracking'})

        # Create figure with 4 subplots in a row
        fig, axes = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)

        # Remove top and right spines from all subplots
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Define a consistent color palette
        colors = ['#FF69B4', '#3498db']  # Pink, Blue

        # Error bar color (for the vertical lines)
        error_bar_color = 'black'
        error_cap_color = 'black'

        # Ensure consistent ordering of categories
        order = ['Backtracking', 'No Backtracking']
        hue_order = order  # Same order for hue to ensure consistent colors

        # Function to perform t-test and return p-value and annotation
        def get_t_test_annotation(data, group_col, value_col, group1, group2):
            group1_data = data[data[group_col] == group1][value_col]
            group2_data = data[data[group_col] == group2][value_col]

            # Only perform t-test if we have enough data
            if len(group1_data) < 2 or len(group2_data) < 2:
                return "N/A", ""

            t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)

            # Calculate the mean difference
            mean1 = group1_data.mean()
            mean2 = group2_data.mean()
            diff = mean1 - mean2

            # Create annotation with significance stars
            if p_val < 0.01:
                stars = "***"  # p < 0.01
            elif p_val < 0.05:
                stars = "**"  # p < 0.05
            elif p_val < 0.1:
                stars = "*"  # p < 0.1
            else:
                stars = ""  # not significant (empty space instead of "ns")

            annotation = f"Δ = {diff:.2f} {stars}"
            return p_val, annotation

        # Helper function to calculate CI
        def get_ci(data):
            import numpy as np
            from scipy import stats

            mean = np.mean(data)
            ci = stats.norm.interval(0.95, loc=mean, scale=stats.sem(data))
            return mean, ci[0], ci[1]

        # Function to add custom error bars with colored caps
        def add_custom_error_bars(ax, data, x_col, y_col, x_positions, color='black', cap_color='#FF69B4'):
            import numpy as np

            for i, x_pos in enumerate(x_positions):
                group_data = data[data[x_col] == order[i]][y_col]

                if len(group_data) < 2:
                    continue

                mean, ci_low, ci_high = get_ci(group_data)

                # Vertical error bar
                ax.plot([x_pos, x_pos], [ci_low, ci_high], color=color, linewidth=2.0)

                # Top cap (horizontal line) with custom color - narrower width
                ax.plot([x_pos - 0.03, x_pos + 0.03], [ci_high, ci_high],
                        color=cap_color, linewidth=1.5, marker='_', markersize=6)

                # Bottom cap (horizontal line) with custom color - narrower width
                ax.plot([x_pos - 0.03, x_pos + 0.03], [ci_low, ci_low],
                        color=cap_color, linewidth=1.5, marker='_', markersize=6)

        # Plot 1: All difficulty levels - Use manual positioning for bars to reduce space
        # First create a placeholder barplot to set up axes but don't show it
        sns.barplot(
            data=df,
            x='backtrack_label',
            y='is_correct',
            hue='backtrack_label',
            order=order,
            hue_order=hue_order,
            ax=axes[0],
            palette=colors,
            errorbar=None,
            legend=False,
            alpha=0.0,  # Make invisible
            width=0.4  # Keep original width
        )

        # Remove the invisible bars
        for patch in axes[0].patches:
            patch.remove()

        # Define custom positions for bars - making them closer together
        positions = [0.2, 0.8]  # Custom x-positions that are closer together
        bar_width = 0.4

        # Manually add bars at custom positions
        for i, (label, color) in enumerate(zip(order, colors)):
            data = df[df['backtrack_label'] == label]['is_correct']
            height = data.mean()
            axes[0].bar(
                positions[i],
                height,
                width=bar_width,
                color=color,
                alpha=0.6,
                label=label
            )

        # Ensure proper centering of x-axis labels
        axes[0].set_xticks(positions)
        axes[0].set_xticklabels(order, ha='center')
        axes[0].tick_params(axis='x', pad=10)  # Add some padding for better spacing

        axes[0].set_title('All Difficulty Levels', fontsize=14)
        axes[0].set_ylim(0, 1.2)  # Increased to show full error bars
        axes[0].set_ylabel('Success Rate', fontsize=12)
        axes[0].set_xlabel('')

        # Add custom error bars at the new positions
        add_custom_error_bars(
            axes[0], df, 'backtrack_label', 'is_correct',
            x_positions=positions, color=error_bar_color, cap_color=error_cap_color
        )

        # Add value labels to the left side of all bars
        for i, (label, pos) in enumerate(zip(order, positions)):
            height = df[df['backtrack_label'] == label]['is_correct'].mean()

            # Position all text labels to the left of their respective bars
            axes[0].text(
                pos - 0.07,  # Position to the left of the bar
                height + 0.01,  # Position just above the bar
                f'{height:.2f}',
                ha='right',  # Right-align
                va='bottom',  # Align to bottom
                fontsize=10
            )

        # Add t-test for All Levels - position it between the custom bars
        p_val, annotation = get_t_test_annotation(df, 'backtrack_label', 'is_correct', 'Backtracking',
                                                  'No Backtracking')
        if annotation != "N/A":
            # Position the annotation between the two bars (average of positions)
            middle_pos = (positions[0] + positions[1]) / 2
            axes[0].text(middle_pos, 1.1, annotation, ha='center', fontsize=12)

        # Plot 2-4: Individual difficulty levels
        for i, level in enumerate([1, 2, 3], start=1):
            level_data = df[df['difficulty_level'] == level]

            if len(level_data) > 0:  # Only plot if we have data for this difficulty level
                # Create invisible placeholder plot
                sns.barplot(
                    data=level_data,
                    x='backtrack_label',
                    y='is_correct',
                    hue='backtrack_label',
                    order=order,
                    hue_order=hue_order,
                    ax=axes[i],
                    palette=colors,
                    errorbar=None,
                    legend=False,
                    alpha=0.0,  # Make invisible
                    width=0.4  # Keep original width
                )

                # Remove the invisible bars
                for patch in axes[i].patches:
                    patch.remove()

                # Manually add bars at custom positions
                for j, (label, color) in enumerate(zip(order, colors)):
                    level_label_data = level_data[level_data['backtrack_label'] == label]['is_correct']
                    if len(level_label_data) > 0:
                        height = level_label_data.mean()
                        axes[i].bar(
                            positions[j],
                            height,
                            width=bar_width,
                            color=color,
                            alpha=0.6,
                            label=label
                        )

                # Configure x-axis ticks and labels for better centering
                axes[i].set_xticks(positions)
                axes[i].set_xticklabels(order, ha='center')
                axes[i].tick_params(axis='x', pad=10)  # Add padding

                axes[i].set_title(f'Difficulty Level {level}', fontsize=14)
                axes[i].set_ylim(0, 1.2)  # Increased to show full error bars
                axes[i].set_ylabel('')
                axes[i].set_xlabel('')

                # Add custom error bars at the new positions
                add_custom_error_bars(
                    axes[i], level_data, 'backtrack_label', 'is_correct',
                    x_positions=positions, color=error_bar_color, cap_color=error_cap_color
                )

                # Add value labels to the left side of all bars
                for j, (label, pos) in enumerate(zip(order, positions)):
                    label_data = level_data[level_data['backtrack_label'] == label]['is_correct']
                    if len(label_data) > 0:
                        height = label_data.mean()

                        # Position all text labels to the left of their respective bars
                        axes[i].text(
                            pos - 0.07,  # Position to the left of the bar
                            height + 0.01,  # Position just above the bar
                            f'{height:.2f}',
                            ha='right',  # Right-align
                            va='bottom',  # Align to bottom
                            fontsize=10
                        )

                # Add t-test for this difficulty level
                p_val, annotation = get_t_test_annotation(
                    level_data, 'backtrack_label', 'is_correct', 'Backtracking', 'No Backtracking'
                )
                if annotation != "N/A":
                    # Position the annotation between the two bars
                    middle_pos = (positions[0] + positions[1]) / 2
                    axes[i].text(middle_pos, 1.1, annotation, ha='center', fontsize=12)

            else:
                axes[i].text(0.5, 0.5, f'No data for difficulty level {level}',
                             ha='center', va='center', fontsize=14)
                axes[i].set_title(f'Difficulty Level {level}', fontsize=14)
                axes[i].set_ylim(0, 1.2)  # Keep consistent y-axis

                # Remove spines for empty plots too
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)

        # Add a main title
        fig.suptitle('Success Rate by Backtracking Status and Difficulty Level', fontsize=16, y=1.05)

        # Save or show the figure
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {save_path}")
        else:
            plt.show()

        return fig

    def plot_length_vs_backtracks(self, df: pd.DataFrame, experiment_dir: Path):
        plots_dir = experiment_dir / "plots"
        plt.figure(figsize=(10, 6))
        plt.scatter(df['Length'], df['Num_Backtracks'], alpha=0.5)
        plt.xlabel('Length (characters)')
        plt.ylabel('Number of Backtracks')
        plt.title('Response Length vs Number of Backtracks')

        # Add trend line
        z = np.polyfit(df['Length'], df['Num_Backtracks'], 1)
        p = np.poly1d(z)
        plt.plot(df['Length'], p(df['Length']), "r--", alpha=0.8)

        plt.savefig(plots_dir / "length_vs_backtracks.png")
        plt.close()

    def plot_density_distribution(self, df: pd.DataFrame, experiment_dir: Path):
        plots_dir = experiment_dir / "plots"
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='Backtrack_Density', bins=30)
        plt.xlabel('Backtracks per 1k characters')
        plt.ylabel('Count')
        plt.title('Distribution of Backtrack Density')
        plt.savefig(plots_dir / "backtrack_density.png")
        plt.close()

    def extract_backtrack_features(self, df: pd.DataFrame) -> tuple:
        """Extract core backtracking features for regression analysis."""
        # Initialize the DataFrame based on the input DataFrame
        regression_df = pd.DataFrame(index=df.index)

        # Add basic columns
        regression_df['Benchmark'] = 'GSM8K'
        regression_df['Question'] = df['gsm8k_id']
        regression_df['Success'] = df['is_correct']
        regression_df['Length'] = df['full_response'].str.len()
        regression_df['Has_Backtrack'] = df['backtrack'].astype(int)
        regression_df['Num_Backtracks'] = df['detailed_analysis'].apply(
            lambda x: x['num_backtracks'] if isinstance(x, dict) else 0
        )

        # Add difficulty metrics
        regression_df['num_expressions'] = df['num_expressions']
        regression_df['difficulty_level'] = df['difficulty_level']

        # Add operation type indicators
        regression_df['has_addition'] = df['has_addition']
        regression_df['has_subtraction'] = df['has_subtraction']
        regression_df['has_multiplication'] = df['has_multiplication']
        regression_df['has_division'] = df['has_division']

        # Add density metric
        regression_df['Backtrack_Density'] = (regression_df['Num_Backtracks'] * 1000 /
                                              regression_df['Length'].replace(0, 1))

        # Create a copy without First_Backtrack_Position for saving to CSV
        regression_df_for_csv = regression_df.copy()
        if 'First_Backtrack_Position' in regression_df.columns:
            regression_df_for_csv = regression_df.drop(columns=['First_Backtrack_Position'])

        return regression_df, regression_df_for_csv

    def save_backtrack_features(self, regression_df: pd.DataFrame, regression_df_for_csv: pd.DataFrame,
                                experiment_dir: Path):
        """
        Save the backtrack features to a CSV file.

        Args:
            regression_df: DataFrame with regression features (including First_Backtrack_Position)
            regression_df_for_csv: DataFrame without First_Backtrack_Position
            experiment_dir: Path to the experiment directory
        """
        output_path = experiment_dir / "results" / "main_table.csv"
        regression_df_for_csv.to_csv(output_path, index=False)
        print(f"\nBacktrack features saved to: {output_path}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total samples: {len(regression_df)}")
        print(f"Success rate: {regression_df['Success'].mean():.2%}")
        print(f"Average number of backtracks: {regression_df['Num_Backtracks'].mean():.2f}")
        print(f"Problems with backtracks: {(regression_df['Num_Backtracks'] > 0).sum()}")

        # Print difficulty level summary
        print("\nDifficulty Level Summary:")
        difficulty_counts = regression_df['difficulty_level'].value_counts().sort_index()
        for level, count in difficulty_counts.items():
            print(f"Level {level}: {count} problems ({count / len(regression_df):.2%})")

        # Calculate correlation between backtracks and success
        corr = regression_df['Success'].corr(regression_df['Num_Backtracks'])
        print(f"\nCorrelation between success and number of backtracks: {corr:.3f}")

    # Update the validate_response function to handle potential errors
    def validate_response(self, text: str) -> bool:
        """
        Validate if the response has a proper final answer while ignoring examples.
        """
        try:
            if not isinstance(text, str):
                return False

            text = text.strip()

            # Ignore lines that are clearly examples or instructions
            example_patterns = [
                r'e\.g\.,\s*###\s*ANSWER:',  # Matches e.g., ### ANSWER:
                r'example:?\s*###\s*ANSWER:',  # Matches example: ### ANSWER:
                r'such as\s*###\s*ANSWER:',  # Matches such as ### ANSWER:
                r'format:?\s*###\s*ANSWER:',  # Matches format: ### ANSWER:
                r'write\s+###\s*ANSWER:',  # Matches write ### ANSWER:
                r'you\s+would\s+write\s+###\s*ANSWER:',  # Matches you would write ### ANSWER:
            ]

            # Split text into lines for more precise analysis
            lines = text.split('\n')
            valid_lines = []

            for line in lines:
                # Skip if line matches any example pattern
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in example_patterns):
                    continue
                valid_lines.append(line)

            # Rejoin valid lines
            cleaned_text = '\n'.join(valid_lines)

            # Look for answer patterns in cleaned text
            answer_patterns = [
                r'###\s*ANSWER:\s*-?\d*\.?\d+\s*$',  # ### ANSWER: X at end of line
                r'\*\*Final Answer:?\s*-?\d*\.?\d+\*\*',  # **Final Answer: X**
                r'\*\*Final Answer:?\*\*\s*-?\d*\.?\d+',  # **Final Answer** X
                r'\\boxed\{-?\d*\.?\d+\}',  # \boxed{X}
                r'Therefore,?\s+(?:the\s+)?(?:answer|result)\s+is:?\s*-?\d*\.?\d+\s*$'
                # Therefore the answer is X at end of line
            ]

            for pattern in answer_patterns:
                match = re.search(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        # Try to find matching lines
                        matching_lines = [line for line in valid_lines if match.group() in line]
                        if matching_lines:  # If we found any matching lines
                            # Check if any of the matching lines are not examples
                            for line in matching_lines:
                                if not any(re.search(exp, line, re.IGNORECASE) for exp in example_patterns):
                                    return True
                    except:
                        continue

            return False
        except Exception as e:
            print(f"Error in validate_response: {e}")
            # Return False in case of any errors
            return False

    def extract_model_answer(self, text: str) -> float:
        """
        Extract the model's numerical answer from response while ignoring examples.
        """
        if not isinstance(text, str):
            return None

        text = text.strip()

        # First clean the text by removing example lines
        example_patterns = [
            r'e\.g\.,\s*###\s*ANSWER:',
            r'example:?\s*###\s*ANSWER:',
            r'such as\s*###\s*ANSWER:',
            r'format:?\s*###\s*ANSWER:',
            r'write\s+###\s*ANSWER:',
            r'you\s+would\s+write\s+###\s*ANSWER:'
        ]

        lines = text.split('\n')
        valid_lines = []

        for line in lines:
            if not any(re.search(pattern, line, re.IGNORECASE) for pattern in example_patterns):
                valid_lines.append(line)

        cleaned_text = '\n'.join(valid_lines)

        # Define patterns in order of preference
        patterns = [
            (r'###\s*ANSWER:\s*(-?\d*\.?\d+)\s*$', 1),  # ### ANSWER: X at end of line
            (r'\*\*Final Answer:?\s*(-?\d*\.?\d+)\*\*', 1),  # **Final Answer: X**
            (r'\*\*Final Answer:?\*\*\s*(-?\d*\.?\d+)', 1),  # **Final Answer** X
            (r'\\boxed\{(-?\d*\.?\d+)\}', 1),  # \boxed{X}
            (r'Therefore,?\s+(?:the\s+)?(?:answer|result)\s+is:?\s*(-?\d*\.?\d+)\s*$', 1)
            # Therefore the answer is X at end
        ]

        for pattern, group in patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    # Try to find matching lines
                    matching_lines = [line for line in valid_lines if match.group() in line]
                    if matching_lines:  # If we found any matching lines
                        # Check if any of the matching lines are not examples
                        for line in matching_lines:
                            if not any(re.search(exp, line, re.IGNORECASE) for exp in example_patterns):
                                return float(match.group(group))
                except (ValueError, TypeError):
                    continue

        return None

    def extract_true_answer(self, answer_text: str) -> float:
        """Extract true answer from GSM8K answer"""
        try:
            if not isinstance(answer_text, str):
                return None

            # Look for #### format first
            match = re.search(r'####\s*(-?\d*\.?\d+)', answer_text)
            if match:
                return float(match.group(1))

            # Look for the last <<expression=number>> pattern
            matches = re.finditer(r'<<.*?=\s*(-?\d*\.?\d+)>>', answer_text)
            last_match = None
            for last_match in matches:
                pass
            if last_match:
                return float(last_match.group(1))

            # Look for the last number in the text
            numbers = re.findall(r'-?\d*\.?\d+', answer_text)
            if numbers:
                return float(numbers[-1])

            return None
        except Exception:
            return None

    def setup_experiment_folder(self, input_file: Path) -> Path:
        """Create a new experiment folder with timestamp and copy input file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = self.base_dir / "experiments" / f"analysis_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (experiment_dir / "input").mkdir()
        (experiment_dir / "results").mkdir()
        (experiment_dir / "plots").mkdir()  # New directory for plots

        # Copy input file to experiment directory
        input_copy = experiment_dir / "input" / input_file.name
        shutil.copy2(input_file, input_copy)

        return experiment_dir

    def create_comprehensive_plots(self, df: pd.DataFrame, plots_dir: Path):
        """Create separate visualization files for each analysis component"""
        # Calculate response lengths first
        df['response_length'] = df['full_response'].str.len()

        # 1. Heatmap of success rates across difficulty and backtracking
        plt.figure(figsize=(10, 8))
        success_matrix = pd.pivot_table(
            df,
            values='is_correct',
            index='difficulty_level',
            columns='backtrack',
            aggfunc='mean'
        )
        sns.heatmap(success_matrix,
                    annot=True,
                    fmt='.2%',
                    cmap='YlOrRd')
        plt.title('Success Rate by Difficulty and Backtracking')
        plt.xlabel('Has Backtrack')
        plt.ylabel('Difficulty Level')
        plt.tight_layout()
        plt.savefig(plots_dir / 'success_rate_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Bar plot of overall success rates
        plt.figure(figsize=(10, 8))

        # Calculate mean success rates and confidence intervals
        success_means = []
        confidence_intervals = []
        labels = []

        for has_backtrack in [False, True]:
            mask = df['backtrack'] == has_backtrack
            success_rate = df[mask]['is_correct'].mean()
            std = df[mask]['is_correct'].std()
            n = len(df[mask])
            ci = 1.96 * (std / np.sqrt(n))

            success_means.append(success_rate)
            confidence_intervals.append(ci)
            labels.append('With Backtrack' if has_backtrack else 'Without Backtrack')

        # Create bar plot
        bars = plt.bar(labels, success_means)

        # Add error bars
        plt.errorbar(x=range(len(labels)), y=success_means, yerr=confidence_intervals,
                     fmt='none', color='black', capsize=5)

        plt.title('Overall Success Rate by Backtracking')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(plots_dir / 'overall_success_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Line plot of success rate by difficulty
        plt.figure(figsize=(10, 8))
        success_by_diff = df.groupby(['difficulty_level', 'backtrack'])['is_correct'].mean().unstack()
        success_by_diff.plot(marker='o')
        plt.title('Success Rate by Difficulty Level')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Success Rate')
        plt.legend(['No Backtrack', 'Has Backtrack'])
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'success_by_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Scatter plot of length vs success, colored by backtracking
        plt.figure(figsize=(12, 8))
        colors = ['red', 'blue']
        for i, has_backtrack in enumerate([False, True]):
            mask = df['backtrack'] == has_backtrack
            plt.scatter(
                df[mask]['response_length'],
                df[mask]['is_correct'],
                c=colors[i],
                alpha=0.5,
                label=f'{"With" if has_backtrack else "Without"} Backtrack'
            )

        # Add trend lines
        for i, has_backtrack in enumerate([False, True]):
            mask = df['backtrack'] == has_backtrack
            if sum(mask) > 0:  # Only add trend line if we have data points
                z = np.polyfit(df[mask]['response_length'], df[mask]['is_correct'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df['response_length'].min(), df['response_length'].max(), 100)
                plt.plot(x_trend, p(x_trend), c=colors[i], linestyle='--', alpha=0.8)

        plt.title('Success vs Response Length')
        plt.xlabel('Response Length (characters)')
        plt.ylabel('Success')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'success_vs_length.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. New plot: Difficulty comparison with backtracking/no backtracking
        # Create the composite difficulty comparison plot (all levels and each level separately)
        create_difficulty_comparison_plot(df, plots_dir / 'difficulty_comparison.png')

    def create_difficulty_level_plots(self, df: pd.DataFrame, plots_dir: Path):
        """Create separate visualizations for difficulty level analysis"""

        # 1. Bar chart of difficulty level distribution
        plt.figure(figsize=(10, 6))
        difficulty_counts = df['difficulty_level'].value_counts().sort_index()
        labels = [f"Level {i}" for i in difficulty_counts.index]

        plt.bar(labels, difficulty_counts.values)
        plt.title('Distribution of Problem Difficulty Levels')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Number of Problems')

        # Add count labels on top of bars
        for i, count in enumerate(difficulty_counts.values):
            plt.annotate(str(count),
                         xy=(i, count),
                         ha='center',
                         va='bottom')

        plt.tight_layout()
        plt.savefig(plots_dir / 'difficulty_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Success rate by difficulty level
        plt.figure(figsize=(10, 6))
        success_by_diff = df.groupby('difficulty_level')['is_correct'].mean()

        plt.bar(labels, success_by_diff.values)
        plt.title('Success Rate by Difficulty Level')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)

        # Add percentage labels on top of bars
        for i, rate in enumerate(success_by_diff.values):
            plt.annotate(f"{rate:.1%}",
                         xy=(i, rate),
                         ha='center',
                         va='bottom')

        plt.tight_layout()
        plt.savefig(plots_dir / 'success_by_difficulty_level.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Backtrack rate by difficulty level
        plt.figure(figsize=(10, 6))
        backtrack_by_diff = df.groupby('difficulty_level')['backtrack'].mean()

        plt.bar(labels, backtrack_by_diff.values)
        plt.title('Backtrack Rate by Difficulty Level')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Backtrack Rate')
        plt.ylim(0, 1)

        # Add percentage labels on top of bars
        for i, rate in enumerate(backtrack_by_diff.values):
            plt.annotate(f"{rate:.1%}",
                         xy=(i, rate),
                         ha='center',
                         va='bottom')

        plt.tight_layout()
        plt.savefig(plots_dir / 'backtrack_by_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Operation type distribution
        plt.figure(figsize=(10, 6))
        operation_data = {
            'Addition': df['has_addition'].sum(),
            'Subtraction': df['has_subtraction'].sum(),
            'Multiplication': df['has_multiplication'].sum(),
            'Division': df['has_division'].sum()
        }

        plt.bar(operation_data.keys(), operation_data.values())
        plt.title('Distribution of Operation Types')
        plt.xlabel('Operation Type')
        plt.ylabel('Number of Problems')

        # Add count labels on top of bars
        for i, (op, count) in enumerate(operation_data.items()):
            plt.annotate(str(count),
                         xy=(i, count),
                         ha='center',
                         va='bottom')

        plt.tight_layout()
        plt.savefig(plots_dir / 'operation_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Success rate by operation type presence (as a separate plot)
        plt.figure(figsize=(10, 6))
        operation_success = {
            'Addition': df[df['has_addition']]['is_correct'].mean(),
            'Subtraction': df[df['has_subtraction']]['is_correct'].mean(),
            'Multiplication': df[df['has_multiplication']]['is_correct'].mean(),
            'Division': df[df['has_division']]['is_correct'].mean()
        }

        plt.bar(operation_success.keys(), operation_success.values())
        plt.title('Success Rate by Operation Type')
        plt.xlabel('Operation Type')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)

        # Add percentage labels
        for i, (op, rate) in enumerate(operation_success.items()):
            plt.annotate(f"{rate:.1%}",
                         xy=(i, rate),
                         ha='center',
                         va='bottom')

        plt.tight_layout()
        plt.savefig(plots_dir / 'operation_success_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_results(self, input_file: Path):
        """Main analysis function"""
        try:
            # Setup experiment directory
            experiment_dir = self.setup_experiment_folder(input_file)
            print(f"\nCreated experiment directory: {experiment_dir}")

            # Read data
            df = pd.read_csv(input_file)
            initial_count = len(df)

            # Filter out incomplete or improperly formatted responses
            print("Validating responses...")
            df['is_valid'] = df['full_response'].apply(self.validate_response)
            df_filtered = df[df['is_valid']].copy()
            filtered_count = len(df_filtered)

            print(f"\nFiltering responses:")
            print(f"Initial responses: {initial_count}")
            print(f"Valid responses: {filtered_count}")
            print(f"Filtered out: {initial_count - filtered_count} responses")

            if filtered_count == 0:
                raise ValueError("No valid responses found after filtering!")

            # Add analysis columns to filtered data
            print("\nAnalyzing responses...")
            df_filtered['backtrack'] = df_filtered['full_response'].apply(
                lambda x: bool(self.backtrack_analyzer.backtrack_regex.search(x)))
            df_filtered['extracted_answer'] = df_filtered['full_response'].apply(self.extract_model_answer)
            df_filtered['true_answer'] = df_filtered['answer'].apply(self.extract_true_answer)

            # Check correctness with tolerance
            df_filtered['is_correct'] = df_filtered.apply(
                lambda row: 1 if row['extracted_answer'] is not None and
                                 row['true_answer'] is not None and
                                 abs(row['extracted_answer'] - row['true_answer']) < 0.01
                else 0, axis=1
            )

            # Perform detailed backtrack analysis
            print("Performing detailed backtrack analysis...")
            detailed_analysis = self.backtrack_analyzer.analyze_batch(df_filtered)
            df_filtered['detailed_analysis'] = detailed_analysis

            # Create DifficultyAnalyzer instance
            difficulty_analyzer = DifficultyAnalyzer()

            # Calculate difficulty using the new operation-based approach
            print("Calculating problem difficulty using operation-based approach...")
            df_filtered = difficulty_analyzer.batch_analyze(df_filtered, answer_column='answer')

            # Calculate base statistics
            stats = self.calculate_stats(df_filtered)

            # Add backtracking stats
            backtrack_stats = self.backtrack_analyzer.generate_stats(detailed_analysis)
            stats.update({"backtrack_analysis": backtrack_stats})

            # Pattern analysis
            print("Analyzing patterns...")
            pattern_analyzer = PatternAnalyzer()
            pattern_results = []
            for analysis in detailed_analysis:
                if analysis['detailed_steps']:
                    patterns = pattern_analyzer.analyze_patterns(analysis['detailed_steps'])
                    pattern_results.append(patterns)

            # Generate pattern statistics and update stats
            pattern_stats = pattern_analyzer.generate_pattern_stats(pattern_results)
            stats.update({"pattern_analysis": pattern_stats})

            # Generate backtrack features
            print("Generating backtrack features...")
            backtrack_df, backtrack_df_for_csv = self.extract_backtrack_features(df_filtered)
            self.save_backtrack_features(backtrack_df, backtrack_df_for_csv, experiment_dir)

            # Create plots directory if it doesn't exist
            plots_dir = experiment_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Optional: Add visualizations
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Success', y='Num_Backtracks', data=backtrack_df)
            plt.title('Number of Backtracks vs Success')
            plt.savefig(plots_dir / "backtracks_vs_success.png")
            plt.close()

            # Create new density plots
            self.create_density_plots(df_filtered, plots_dir)

            # Generate plots from backtrack analyzer
            self.backtrack_analyzer.plot_backtrack_distribution(
                detailed_analysis,
                plots_dir / "backtrack_distribution.png"
            )

            self.backtrack_analyzer.plot_accuracy_comparison(
                detailed_analysis,
                plots_dir / "accuracy_comparison.png"
            )

            self.backtrack_analyzer.plot_operations_distribution(
                detailed_analysis,
                plots_dir / "operations_distribution.png"
            )

            pattern_analyzer.plot_pattern_distribution(
                pattern_stats,
                plots_dir / "pattern_distribution.png"
            )

            # Create new difficulty level based visualizations
            self.create_comprehensive_plots(df_filtered, plots_dir)
            self.create_difficulty_level_plots(df_filtered, plots_dir)

            # Create the new difficulty comparison plot
            create_difficulty_comparison_plot(df_filtered, plots_dir / 'difficulty_comparison.png')

            # Save results
            print("\nSaving results...")
            self.save_results(df, df_filtered, stats, experiment_dir)

            return df_filtered, stats, experiment_dir

        except Exception as e:
            print(f"Error analyzing results: {e}")
            raise

    def calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate analysis statistics"""
        total_problems = len(df)
        total_correct = int(df['is_correct'].sum())
        total_backtracks = sum(1 for x in df['detailed_analysis'] if x['num_backtracks'] > 0)

        stats = {
            "total_problems": total_problems,
            "total_correct": total_correct,
            "accuracy": (total_correct / total_problems * 100) if total_problems > 0 else 0,
            "total_backtracks": total_backtracks,
            "backtrack_rate": (total_backtracks / total_problems * 100) if total_problems > 0 else 0,
            "correct_with_backtrack": int(len(df[(df['backtrack'] == True) & (df['is_correct'] == 1)])),
            "incorrect_with_backtrack": int(len(df[(df['backtrack'] == True) & (df['is_correct'] == 0)])),
            "correct_without_backtrack": int(len(df[(df['backtrack'] == False) & (df['is_correct'] == 1)])),
            "incorrect_without_backtrack": int(len(df[(df['backtrack'] == False) & (df['is_correct'] == 0)]))
        }

        # Add detailed accuracy stats
        if total_backtracks > 0:
            stats["accuracy_with_backtrack"] = (stats["correct_with_backtrack"] / total_backtracks * 100)
        if (total_problems - total_backtracks) > 0:
            stats["accuracy_without_backtrack"] = (stats["correct_without_backtrack"] /
                                                   (total_problems - total_backtracks) * 100)

        # Add difficulty level statistics
        difficulty_stats = {}
        for level in sorted(df['difficulty_level'].unique()):
            level_df = df[df['difficulty_level'] == level]
            difficulty_stats[f"level_{level}"] = {
                "count": len(level_df),
                "correct": int(level_df['is_correct'].sum()),
                "accuracy": (level_df['is_correct'].mean() * 100),
                "backtrack_rate": (level_df['backtrack'].mean() * 100)
            }

        stats["difficulty_analysis"] = difficulty_stats

        return stats

    def create_density_plots(self, df: pd.DataFrame, plots_dir: Path):
        """Create new visualizations for backtrack density analysis"""
        plots_dir.mkdir(exist_ok=True)

        # Calculate number of backtracks and length
        backtracks = df['detailed_analysis'].apply(lambda x: x['num_backtracks'] if isinstance(x, dict) else 0)
        lengths = df['full_response'].str.len()

        # Split data into successful and unsuccessful cases
        success_mask = df['is_correct'] == 1

        # Create length vs backtracks scatter plot with different markers
        plt.figure(figsize=(12, 8))

        # Plot unsuccessful cases with 'x' markers in red
        plt.scatter(lengths[~success_mask], backtracks[~success_mask],
                    marker='x', color='red', alpha=0.6, label='Incorrect',
                    s=100)  # Increased marker size for better visibility

        # Plot successful cases with 'o' markers in green
        plt.scatter(lengths[success_mask], backtracks[success_mask],
                    marker='o', color='green', alpha=0.6, label='Correct',
                    s=100)  # Increased marker size for better visibility

        plt.xlabel('Length (characters)')
        plt.ylabel('Number of Backtracks')
        plt.title('Response Length vs Number of Backtracks')

        # Add trend line for all data
        z = np.polyfit(lengths, backtracks, 1)
        p = np.poly1d(z)
        plt.plot(lengths, p(lengths), "--", color='blue', alpha=0.8, label='Trend Line')

        # Add legend
        plt.legend()

        # Add grid for better readability
        plt.grid(True, alpha=0.3)

        plt.savefig(plots_dir / "length_vs_backtracks.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate and plot density
        density = backtracks * 1000 / lengths

        plt.figure(figsize=(10, 6))
        sns.histplot(data=density, bins=30)
        plt.xlabel('Backtracks per 1k characters')
        plt.ylabel('Count')
        plt.title('Distribution of Backtrack Density')
        plt.savefig(plots_dir / "backtrack_density.png")
        plt.close()

    def calculate_density_stats(self, df: pd.DataFrame) -> dict:
        """Calculate statistics for backtrack density"""
        backtracks = df['detailed_analysis'].apply(lambda x: x['num_backtracks'] if isinstance(x, dict) else 0)
        lengths = df['full_response'].str.len()
        density = backtracks * 1000 / lengths

        return {
            'mean_density': float(density.mean()),
            'median_density': float(density.median()),
            'std_density': float(density.std()),
            'min_density': float(density.min()),
            'max_density': float(density.max())
        }

    def save_results(self, df_original: pd.DataFrame, df_filtered: pd.DataFrame,
                     stats: dict, experiment_dir: Path):
        """Save analysis results in the experiment directory"""
        results_dir = experiment_dir / "results"
        plots_dir = experiment_dir / "plots"

        # Save filtered analysis results
        analysis_path = results_dir / "detailed_analysis.csv"
        df_filtered_save = df_filtered.drop(columns=['detailed_analysis'])
        df_filtered_save.to_csv(analysis_path, index=False)

        # Save invalid responses separately
        invalid_responses = df_original[~df_original['is_valid']].copy()
        invalid_path = results_dir / "invalid_responses.csv"
        invalid_responses.to_csv(invalid_path, index=False)

        # Save detailed backtrack analysis separately
        backtrack_path = results_dir / "backtrack_analysis.json"
        with open(backtrack_path, 'w') as f:
            detailed_results = []
            for idx, row in df_filtered.iterrows():
                result = {
                    'gsm8k_id': row['gsm8k_id'],
                    'is_correct': row['is_correct'],
                    'backtrack_analysis': row['detailed_analysis']
                }
                # Add difficulty and operation features if available
                for col in ['difficulty_level', 'has_addition', 'has_subtraction',
                            'has_multiplication', 'has_division']:
                    if col in row:
                        result[col] = row[col]
                detailed_results.append(result)
            json.dump(detailed_results, f, indent=2)

        # Create and save density plots
        self.create_density_plots(df_filtered, plots_dir)

        # Calculate density statistics
        density_stats = self.calculate_density_stats(df_filtered)
        stats['density_analysis'] = density_stats

        # Try to create comprehensive plots if the methods exist
        try:
            if hasattr(self, 'create_comprehensive_plots'):
                self.create_comprehensive_plots(df_filtered, plots_dir)

            if hasattr(self, 'create_difficulty_level_plots'):
                self.create_difficulty_level_plots(df_filtered, plots_dir)
        except Exception as e:
            print(f"Warning: Could not create some plots: {e}")

        # Save all statistics
        stats_path = results_dir / "analysis_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Generate original report
        self.generate_report(df_filtered, invalid_responses, stats, results_dir)

        print(f"\nResults saved in: {results_dir}")
        print(f"- Filtered analysis: {analysis_path}")
        print(f"- Invalid responses: {invalid_path}")
        print(f"- Backtrack analysis: {backtrack_path}")
        print(f"- Statistics: {stats_path}")
        print(f"- Plots directory: {plots_dir}")

    def generate_report(self, df_filtered: pd.DataFrame, df_invalid: pd.DataFrame,
                        stats: dict, results_dir: Path):
        """Generate detailed analysis report"""
        report_path = results_dir / "analysis_report.txt"

        with open(report_path, 'w') as f:
            f.write("GSM8K Analysis Report\n====================\n\n")

            # Data filtering statistics
            f.write("Data Filtering:\n")
            f.write(f"Initial samples: {stats['total_problems'] + len(df_invalid)}\n")
            f.write(f"Valid samples: {stats['total_problems']}\n")
            f.write(f"Filtered out: {len(df_invalid)}\n")
            f.write(
                f"Completion rate: {(stats['total_problems'] / (stats['total_problems'] + len(df_invalid)) * 100):.2f}%\n\n")

            # Overall statistics
            f.write("Performance Statistics:\n")
            f.write(f"Correct answers: {stats['total_correct']}\n")
            f.write(f"Overall accuracy: {stats['accuracy']:.2f}%\n\n")

            # Difficulty level analysis
            f.write("Difficulty Level Analysis:\n")
            for level, level_stats in stats.get('difficulty_analysis', {}).items():
                level_num = level.split('_')[1]
                f.write(f"Level {level_num}:\n")
                f.write(f"  Problems: {level_stats['count']}\n")
                f.write(f"  Correct: {level_stats['correct']}\n")
                f.write(f"  Accuracy: {level_stats['accuracy']:.2f}%\n")
                f.write(f"  Backtrack rate: {level_stats['backtrack_rate']:.2f}%\n")
            f.write("\n")

            # Backtracking analysis
            f.write("Backtracking Analysis:\n")
            f.write(f"Problems with backtracking: {stats['total_backtracks']}\n")
            f.write(f"Backtracking rate: {stats['backtrack_rate']:.2f}%\n")

            if 'backtrack_analysis' in stats:
                ba = stats['backtrack_analysis']
                f.write("\nDetailed Backtracking Statistics:\n")
                f.write(f"Average backtracks per solution: {ba['avg_backtracks_per_solution']:.2f}\n")
                f.write("\nBacktrack Positions:\n")
                f.write(f"Early (0-33%): {ba['backtrack_positions']['early']}\n")
                f.write(f"Middle (33-66%): {ba['backtrack_positions']['middle']}\n")
                f.write(f"Late (66-100%): {ba['backtrack_positions']['late']}\n")

                if 'operations_with_backtrack' in ba:
                    f.write("\nOperations with Backtracking:\n")
                    for op, count in ba['operations_with_backtrack'].items():
                        f.write(f"{op}: {count}\n")

            # Density analysis
            if 'density_analysis' in stats:
                f.write("\nBacktrack Density Analysis:\n")
                density = stats['density_analysis']
                f.write(f"Average backtracks per 1k chars: {density['mean_density']:.2f}\n")
                f.write(f"Median density: {density['median_density']:.2f}\n")
                f.write(f"Density std dev: {density['std_density']:.2f}\n")

            # Accuracy comparison
            if 'accuracy_with_backtrack' in stats:
                f.write(f"\nAccuracy with backtracking: {stats['accuracy_with_backtrack']:.2f}%\n")
            if 'accuracy_without_backtrack' in stats:
                f.write(f"Accuracy without backtracking: {stats['accuracy_without_backtrack']:.2f}%\n")

            # Operation type distribution
            operation_counts = {
                'Addition': df_filtered['has_addition'].sum(),
                'Subtraction': df_filtered['has_subtraction'].sum(),
                'Multiplication': df_filtered['has_multiplication'].sum(),
                'Division': df_filtered['has_division'].sum()
            }

            f.write("\nOperation Type Distribution:\n")
            for op, count in operation_counts.items():
                f.write(f"{op}: {count} problems ({count / len(df_filtered):.2f}%)\n")




def main():
    """Main function to coordinate all analysis steps"""
    try:
        # Initialize analyzers
        result_analyzer = ResultAnalyzer()
        difficulty_analyzer = DifficultyAnalyzer()

        # Find most recent CSV
        input_files = list(result_analyzer.input_dir.glob("*.csv"))
        if not input_files:
            raise FileNotFoundError(f"No CSV files found in: {result_analyzer.input_dir}")

        latest_input = max(input_files, key=lambda x: x.stat().st_mtime)
        print(f"\nAnalyzing: {latest_input.name}")

        # Setup experiment directory
        experiment_dir = result_analyzer.setup_experiment_folder(latest_input)
        print(f"\nCreated experiment directory: {experiment_dir}")

        # Read and validate data
        print("Reading and validating data...")
        df = pd.read_csv(latest_input)
        initial_count = len(df)

        # Apply validation and handle potential NaNs explicitly
        validation_results = df['full_response'].apply(result_analyzer.validate_response)

        # Check for NaN values
        if validation_results.isna().any():
            print(f"Found {validation_results.isna().sum()} NaN values in validation results. Converting to False.")
            validation_results = validation_results.fillna(False)

        df['is_valid'] = validation_results

        # Make sure is_valid is boolean
        df['is_valid'] = df['is_valid'].astype(bool)

        # Now we can safely filter
        df_filtered = df[df['is_valid']].copy()
        filtered_count = len(df_filtered)

        print(f"\nFiltering responses:")
        print(f"Initial responses: {initial_count}")
        print(f"Valid responses: {filtered_count}")
        print(f"Filtered out: {initial_count - filtered_count} responses")

        if filtered_count == 0:
            raise ValueError("No valid responses found after filtering!")

        # Perform backtrack analysis
        print("\nAnalyzing backtracking patterns...")
        df_filtered['backtrack'] = df_filtered['full_response'].apply(
            lambda x: bool(result_analyzer.backtrack_analyzer.backtrack_regex.search(x)))
        df_filtered['extracted_answer'] = df_filtered['full_response'].apply(result_analyzer.extract_model_answer)
        df_filtered['true_answer'] = df_filtered['answer'].apply(result_analyzer.extract_true_answer)

        # Check correctness
        df_filtered['is_correct'] = df_filtered.apply(
            lambda row: 1 if row['extracted_answer'] is not None and
                             row['true_answer'] is not None and
                             abs(row['extracted_answer'] - row['true_answer']) < 0.01
            else 0, axis=1
        )

        # Detailed backtrack analysis
        print("Performing detailed backtrack analysis...")
        detailed_analysis = result_analyzer.backtrack_analyzer.analyze_batch(df_filtered)
        df_filtered['detailed_analysis'] = detailed_analysis

        # Calculate difficulty scores
        print("Calculating problem difficulty...")
        df_filtered = difficulty_analyzer.batch_analyze(df_filtered, answer_column='answer')

        # Add these print statements here:
        print("\nDifficulty level distribution:")
        print(df_filtered['difficulty_level'].value_counts().sort_index())
        print("\nAvailable columns:")
        print(df_filtered.columns)

        # Generate all stats
        stats = result_analyzer.calculate_stats(df_filtered)
        backtrack_stats = result_analyzer.backtrack_analyzer.generate_stats(detailed_analysis)
        stats.update({"backtrack_analysis": backtrack_stats})

        # Pattern analysis
        print("Analyzing patterns...")
        pattern_analyzer = PatternAnalyzer()
        pattern_results = []
        for analysis in detailed_analysis:
            if analysis['detailed_steps']:
                patterns = pattern_analyzer.analyze_patterns(analysis['detailed_steps'])
                pattern_results.append(patterns)

        pattern_stats = pattern_analyzer.generate_pattern_stats(pattern_results)
        stats.update({"pattern_analysis": pattern_stats})

        # Generate regression features
        print("Generating regression features...")
        backtrack_df, backtrack_df_for_csv = result_analyzer.extract_backtrack_features(df_filtered)

        # Create visualizations
        print("Generating visualizations...")
        plots_dir = experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        result_analyzer.create_density_plots(df_filtered, plots_dir)
        result_analyzer.backtrack_analyzer.plot_backtrack_distribution(
            detailed_analysis,
            plots_dir / "backtrack_distribution.png"
        )
        result_analyzer.backtrack_analyzer.plot_accuracy_comparison(
            detailed_analysis,
            plots_dir / "accuracy_comparison.png"
        )
        result_analyzer.backtrack_analyzer.plot_operations_distribution(
            detailed_analysis,
            plots_dir / "operations_distribution.png"
        )
        pattern_analyzer.plot_pattern_distribution(
            pattern_stats,
            plots_dir / "pattern_distribution.png"
        )
        # Create the new difficulty comparison plot
        result_analyzer.create_difficulty_comparison_plot(
            df_filtered,
            plots_dir / "difficulty_comparison.png"
        )

        # Save all results
        print("\nSaving results...")
        result_analyzer.save_results(df, df_filtered, stats, experiment_dir)
        result_analyzer.save_backtrack_features(backtrack_df, backtrack_df_for_csv, experiment_dir)

        print("\nAnalysis completed successfully!")
        print(f"\nResults available in: {experiment_dir}")

        return df_filtered, stats, experiment_dir

    except Exception as e:
        print(f"Error in analysis: {e}")
        raise

if __name__ == "__main__":
    main()
