#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define POP_SIZE 100
#define GENE_SIZE 10
#define MUTATION_RATE 5

typedef struct {
    int genes[GENE_SIZE];
    int fitness;
} Individual;

int target_genes[GENE_SIZE] = {65, 73, 95, 69, 86, 79, 76, 86, 69, 83};

void calculate_fitness(Individual *ind) {
    ind->fitness = 0;
    for (int i = 0; i < GENE_SIZE; i++) {
        if (ind->genes[i] == target_genes[i]) {
            ind->fitness++;
        }
    }
}

void initialize_population(Individual *pop) {
    for (int i = 0; i < POP_SIZE; i++) {
        for (int j = 0; j < GENE_SIZE; j++) {
            pop[i].genes[j] = (rand() % 58) + 65;
        }
        calculate_fitness(&pop[i]);
    }
}

int select_parent(Individual *pop) {
    int tournament_size = 5;
    int best_idx = rand() % POP_SIZE;
    for (int i = 1; i < tournament_size; i++) {
        int candidate_idx = rand() % POP_SIZE;
        if (pop[candidate_idx].fitness > pop[best_idx].fitness) {
            best_idx = candidate_idx;
        }
    }
    return best_idx;
}

void crossover(Individual *p1, Individual *p2, Individual *c1, Individual *c2) {
    int midpoint = rand() % GENE_SIZE;
    for (int i = 0; i < GENE_SIZE; i++) {
        if (i < midpoint) {
            c1->genes[i] = p1->genes[i];
            c2->genes[i] = p2->genes[i];
        } else {
            c1->genes[i] = p2->genes[i];
            c2->genes[i] = p1->genes[i];
        }
    }
}

void mutate(Individual *ind) {
    for (int i = 0; i < GENE_SIZE; i++) {
        if ((rand() % 100) < MUTATION_RATE) {
            ind->genes[i] = (rand() % 58) + 65;
        }
    }
}

int main(void) {
    srand((unsigned int)time(NULL));
    
    Individual population[POP_SIZE];
    Individual next_generation[POP_SIZE];
    
    initialize_population(population);
    
    int generation = 0;
    int max_generations = 1000;
    int perfect_score = GENE_SIZE;
    int found = 0;
    
    while (generation < max_generations && !found) {
        int best_fitness = -1;
        int best_idx = 0;
        
        for (int i = 0; i < POP_SIZE; i++) {
            if (population[i].fitness > best_fitness) {
                best_fitness = population[i].fitness;
                best_idx = i;
            }
        }
        
        putchar(71);
        putchar(101);
        putchar(110);
        putchar(58);
        putchar(32);
        
        int gen_temp = generation;
        if (gen_temp == 0) {
            putchar(48);
        } else {
            int digits[10];
            int d_count = 0;
            while (gen_temp > 0) {
                digits[d_count++] = (gen_temp % 10) + 48;
                gen_temp /= 10;
            }
            for (int i = d_count - 1; i >= 0; i--) {
                putchar(digits[i]);
            }
        }
        
        putchar(32);
        putchar(124);
        putchar(32);
        
        for (int j = 0; j < GENE_SIZE; j++) {
            putchar(population[best_idx].genes[j]);
        }
        putchar(10);
        
        if (best_fitness == perfect_score) {
            found = 1;
            break;
        }
        
        for (int i = 0; i < POP_SIZE; i += 2) {
            int p1_idx = select_parent(population);
            int p2_idx = select_parent(population);
            
            crossover(&population[p1_idx], &population[p2_idx], &next_generation[i], &next_generation[i+1]);
            
            mutate(&next_generation[i]);
            mutate(&next_generation[i+1]);
            
            calculate_fitness(&next_generation[i]);
            calculate_fitness(&next_generation[i+1]);
        }
        
        for (int i = 0; i < POP_SIZE; i++) {
            population[i] = next_generation[i];
        }
        
        generation++;
    }
    
    return 0;
}
