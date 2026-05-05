#!/bin/bash

# Run a batch of 10 tests using python main.py with dynamic test names and seeds

#### OpenAI ####
for i in {1..10}; do
    test_name="official_test_influence_circle_collision_gpt_$i"
    seed=$i
    echo "Running test $i with name $test_name and seed $seed..."
    python main.py -n "$test_name" -s "$seed" -gpt gpt-5-mini -mc openai -ra medium -form circle -a 10
    if [ $? -ne 0 ]; then
        echo "Test $i ($test_name, seed $seed) failed. Exiting..."
        exit 1
    fi
    echo "Test $i ($test_name, seed $seed) completed successfully."
done

echo "All tests completed successfully."

#
##### Qwen ####
#for i in {1..10}; do
#    test_name="official_test_influence_circle_qwen_$i"
#    seed=$i
#    echo "Running test $i with name $test_name and seed $seed..."
#    python main.py -n "$test_name" -s "$seed" -gpt qwen-max -mc qwen -form circle -a 10
#    if [ $? -ne 0 ]; then
#        echo "Test $i ($test_name, seed $seed) failed. Exiting..."
#        exit 1
#    fi
#    echo "Test $i ($test_name, seed $seed) completed successfully."
#done
#
##### llama ####
#for i in {1..10}; do
#    test_name="official_test_influence_circle_llama_$i"
#    seed=$i
#    echo "Running test $i with name $test_name and seed $seed..."
#    python main.py -n "$test_name" -s "$seed" -gpt llama3.1-405b -mc llama_api -form circle -a 10
#    if [ $? -ne 0 ]; then
#        echo "Test $i ($test_name, seed $seed) failed. Exiting..."
#        exit 1
#    fi
#    echo "Test $i ($test_name, seed $seed) completed successfully."
#done
#
#### Anthropic ####
# for i in {1..10}; do
#    test_name="official_test_influence_circle_claude_$i"
#    seed=$i
#    echo "Running test $i with name $test_name and seed $seed..."
#    python main.py -n "$test_name" -s "$seed" -gpt claude-sonnet-4-20250514 -mc claude --use_pydantic_ai -form circle -a 10
#    if [ $? -ne 0 ]; then
#        echo "Test $i ($test_name, seed $seed) failed. Exiting..."
#        exit 1
#    fi
#    echo "Test $i ($test_name, seed $seed) completed successfully."
# done
#
##### deepseek ####
# for i in {1..10}; do
#    test_name="official_test_influence_circle_collision_deepseek_$i"
#    seed=$i
#    echo "Running test $i with name $test_name and seed $seed..."
#    python main.py -n "$test_name" -s "$seed" -gpt deepseek-reasoner -mc deepseek_api -form circle -a 10
#    if [ $? -ne 0 ]; then
#        echo "Test $i ($test_name, seed $seed) failed. Exiting..."
#        exit 1
#    fi
#    echo "Test $i ($test_name, seed $seed) completed successfully."
# done

