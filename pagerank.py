import os
import random
import re
import sys
import numpy as np
DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.

    The transition_model should return a dictionary representing the probability distribution over which page a random surfer would visit next, given a corpus of pages, a current page, and a damping factor.

    Steps:
        1) Check if page has outgoing links:
            Yes:
                - Get all pages pl linked to that page
                - Initialize dictionary with all pages that we want to include in the probability distribution (via for loop dictionary comprehension, for key in dict.keys()). This dictionary, prob_distribution, will be what we return at the end.
                - With probability dampening_factor, randomly choose a page which we have linked to
                for linked_page in pl:
                    prob_distribution[linked_page] += damping_factor/len(pl)
                - With probability 1 - damping_factor, choose randomly a page between all pages in the corpus (Use for loop or map/dictionary comprehension)
                for page in prob_distribution.keys():
                    prob_distribution[page] += (1 - damping_factor)/len(prob_distribution)
                - Return probability distribution. assert that all probabilities add up to 1
            No:
                - Pretend it has links to all pages in the corpus, including itself
                - Return probability distribution with all probabilities equal to 1/len(prob_distribution). assert that all probabilities add up to 1
                prob_distribution_final = {key:1/len(prob_distribution) for (key,value) in dictionary.items()}
    """
    #Get all pages pl that page links to
    outgoing_links = corpus[page]
    #Sanity check
    assert corpus[page] != None, f"Cant fetch links from corpus {corpus}! {page} page not found..."

    #initialize prob_distribution variable which is the probability distribution we want to return
    prob_distribution = {key:0 for key in corpus.keys()}

    if outgoing_links != set():
        #With probability damping_factor, randomly choose a page which we have linked to
        for linked_page in outgoing_links:
            prob_distribution[linked_page] += damping_factor/len(outgoing_links)
        
        #With probability 1 - damping_factor, randomly choose a page from the corpus
        for page in prob_distribution.keys():
            prob_distribution[page] += (1 - damping_factor)/len(prob_distribution)
        
        # assert sum(prob_distribution.values()) == 1, f"Error! probabilities {prob_distribution} dont sum up to 1!! current page is {page} and outgoing_links are {outgoing_links}. \n They sum up to {sum(prob_distribution.values())}"
        return prob_distribution
    else:
        #No outgoing links
        assert outgoing_links != "",f"Outgoing_links {outgoing_links} is an empty string!"
        prob_distribution_final = {key:1/len(prob_distribution) for key in prob_distribution.keys()}
        assert sum(prob_distribution_final.values()) == 1,f"Error, while all probabilities are supposed to be equal, probabilities in {prob_distribution_final} dont sum up to 1. \n They sum up to {sum(prob_distribution.values())}"
        return prob_distribution_final




def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.

    Steps:
        1) Choose a page at random (sample = random.choice(list(corpus.keys())))
        2) INitialize counts dictionary with dictionary comprehension (and add one to the page that was chosen initially)
        3)
            for i in range(n - 1):
                #Sample n - 1 more times
                transition_probabilities = transition_model(corpus,sample,damping_factor)
                - Use sample = random.choices(list(transition_model.keys()),k=1,weights=list(transition_model.values()))
                - Add one to the counts dictionary for the page that was chosen
        4) divide counts dictionary by n and return, checking all PRs sum up to 1
    """
    #Choose a page at random
    sample = random.choice(list(corpus.keys()))
    # print(f"First sample is {sample}")

    #Initialize counts dictionary
    counts = {key:0 for key in corpus.keys()}
    counts[sample] += 1

    #For loop
    for i in range(n - 1):
        #Sample n - 1 more times
        transition_probabilities = transition_model(corpus,sample,damping_factor)

        #Pick a new sample based on these probabilities
        sample = random.choices(list(transition_probabilities.keys()),weights=list(transition_probabilities.values()),k=1)[0]
        # print(f"New sample: {sample} chosen! it is a {type(sample)} type")

        #Update counts dictionary
        counts[sample] += 1

    #Calculate final probabilities
    final_probabilities = {key:(value/n) for key,value in counts.items()}
    assert sum(list(final_probabilities.values())) == 1,f"Final probabilites {final_probabilities} dont add up to 1! \n They sum up to {sum(list(final_probabilities.values()))}"
    print(final_probabilities)
    print(sum(list(final_probabilities.values())))
    return final_probabilities


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.

    For the second condition, we need to consider each possible page i that links to page p. For each of those incoming pages, 
    let NumLinks(i) be the number of links on page i. Each page i that links to p has its own PageRank, PR(i),
      representing the probability that we are on page i at any given time. And since from page i we travel to any of that page’s 
      links with equal probability, we divide PR(i) by the number of links NumLinks(i) to get the probability that we were on page i and chose the link to page p.

    Steps:
        N = len(corpus)
        1)Create a page rank dictionary (page_ranks) and assign each page a rank of 1/N where N is total number of pages in corpus
        while max_diff > 0.001:
        #Repeat until convergence
            start_vals = np.array(list(page_ranks.values()))
            for page in corpus.keys():
                2) Find all pages i that link to this page p
                    - Initialize a new set, linked_pages, of all pages i linking to p
                    - Loop through corpus and for every i that links to p, add i to the set
                PR = ((1 - damping_factor)/N) + (damping_factor*sum([(page_ranks[i])/(len(corpus[i])) for i in linked_pages]))
                #Update page rank
                page_ranks[page] = PR
            end_vals = np.array(list(page_ranks.values()))
            max_diff = np.max(np.abs(start_vals-end_vals))

    The iterate_pagerank function should accept a corpus of web pages and a damping factor, calculate PageRanks based on the iteration formula described above, and return each page’s PageRank accurate to within 0.001.

The function accepts two arguments: corpus and damping_factor.
The corpus is a Python dictionary mapping a page name to a set of all pages linked to by that page.
The damping_factor is a floating point number representing the damping factor to be used in the PageRank formula.
The return value of the function should be a Python dictionary with one key for each page in the corpus. Each key should be mapped to a value representing that page’s PageRank. The values in this dictionary should sum to 1.
The function should begin by assigning each page a rank of 1 / N, where N is the total number of pages in the corpus.
The function should then repeatedly calculate new rank values based on all of the current rank values, according to the PageRank formula in the “Background” section. (i.e., calculating a page’s PageRank based on the PageRanks of all pages that link to it).
A page that has no links at all should be interpreted as having one link for every page in the corpus (including itself).
This process should repeat until no PageRank value changes by more than 0.001 between the current rank values and the new rank values.
    """
    #Helper values
    N = len(corpus)
    max_diff = 0.01

    print(f"Old corpus is: {corpus} with {N} elements long")

    #If a page links to no other pages, assume it links to all the pages in the corpus
    corpus_new = {page_name:links if links != set() else set(corpus.keys()) for page_name,links in corpus.items()}
    # print(f"New corpus is: {corpus_new}!",file=f)

    
    #Create page_ranks
    page_ranks = {key:1/N for key in corpus_new.keys()}
    print(f"Initializing page_ranks dict {page_ranks}")

    #Keep iterating While there is a value that changes by more than 0.001. Keep running until there is not a value which changes by more than 0.001
    while max_diff > 0.001:
        #Repeat until convergence
        start_vals = np.array(list(page_ranks.values()))
        for page in corpus_new.keys():
            #Find all pages i that link to this page p
            linked_pages = set()    

            for candidate_page,links in corpus_new.items():
                if page in links:
                    #current candidate_page links to our target page
                    #Add current candidate_page to the linked_pages
                    linked_pages.add(candidate_page)
                    # print("----------------------",file=f)
                    # print(f"Page {page} is linked to by {candidate_page}!",file=f)
                    # print("----------------------------",file=f)
            
            #The function should then repeatedly calculate new rank values based on all of the current rank values, according to the PageRank formula in the “Background” section. (i.e., calculating a page’s PageRank based on the PageRanks of all pages that link to it).
            #A page that has no links at all should be interpreted as having one link for every page in the corpus (including itself).


            #Calculate PR value. Add (1-d)/N plus d * sum(PR(i)/NumLinks(i) for every page i that links to this page p. If the number of links present on page i is 0, then we assume it has a link to every page in the corpus (including itself). )
            PR = ((1 - damping_factor)/N) + (damping_factor*sum([(page_ranks[i])/(len(corpus_new[i])) for i in linked_pages]))
                # print(f"Updating PR to {PR} for page {page}! Pages {linked_pages} link to this page!")


                #Update PR
            page_ranks[page] = PR
        end_vals = np.array(list(page_ranks.values()))

        #Calculate maximum difference
        max_diff = np.max(np.abs(start_vals-end_vals))
            # print(f"Starting PR values are: {start_vals} and ending values are {end_vals} with difference {max_diff}",file=f)
            # print(f"Current page_ranks sums up to: {sum(list(page_ranks.values()))}",file=f)
            # print(f"Current page ranks list is: {page_ranks}",file=f)
            
            # print("==============================================================================================================",file=f)
        
        #Sanity check
        # assert max_diff <= 0.001, f"Convergence requirements not met! max diff {max_diff} with startvalues: {start_vals} and end_values {end_vals} and page_ranks: {page_ranks}"
        # assert sum(list(page_ranks.values())) == 1, f"PRs dont sum up to 1! page_ranks: {page_ranks}, they sum up to {sum(list(page_ranks.values()))}"
    print(sum(list(page_ranks.values())))
    print(page_ranks)

    return page_ranks

if __name__ == "__main__":
    main()
