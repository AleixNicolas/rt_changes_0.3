import csv
import json
import click
import logging
import pandas as pd
import numpy as np
import clustering as cl
import scipy.spatial.distance as  ssd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from twarc import ensure_flattened
from io import TextIOWrapper

logging.getLogger().setLevel(logging.INFO)
    
@click.command()
@click.option('-a', '--alpha', type=click.FLOAT, required=False, default='1')
@click.option('-g', '--granularity', required=False, default=None, type=click.STRING)
@click.option('-t', '--threshold', required=False, default=None, type=click.FLOAT)
@click.option('-i', '--interval', required=False, type=click.STRING)
@click.option('-m', '--method', type=click.STRING, default = 'ward')
@click.option('-l', '--algorithm', type=click.STRING, default = 'generic')
@click.argument('infile1', type=click.File('r'), default='-')
@click.argument('infile2', type=click.File('r'), default='-')

def main(infile1: TextIOWrapper,
         infile2: TextIOWrapper,
         alpha: float,
         threshold: float,
         granularity: str,
         interval: str,
         method: str,
         algorithm: str):

    # Exceptions for algorithm and method    
    if algorithm != 'nn_chain' and algorithm != 'generic':
        raise click.BadOptionUsage('algorithm', 'Non valid algorithm; algorithm must be nn_chain or generic; default is generic')
        
    if algorithm == 'generic' and method != 'centroid' and method != 'poldist' and method != 'ward':
        raise click.BadOptionUsage('method', 'Non valid method; for generic algorithm; method must be poldist, centroid or ward; default is ward')

    if algorithm == 'nn_chain' and method != 'ward':
        raise click.BadOptionUsage('method', 'Non valid method; for nn_chain algorithm; method must be ward; default is ward')
    
    
    #Check for interval validity
    is_interval = False
    start_time = None
    end_time = None

    if interval is not None:
        is_interval = True
        splitted_time = interval.split(',')
        start_time = pd.to_datetime(splitted_time[0], utc=True)
        end_time = pd.to_datetime(splitted_time[1], utc=True)
        
    if alpha !=1:
        total_seconds = int(-3600*24/(np.log(alpha)))
        time_delta =pd.to_timedelta(total_seconds, unit = 'S')
        
    if granularity is not None:
        date_range = pd.date_range(start_time, end_time, freq=granularity)
        for i in range(len(date_range)-1):
            
            logging.info('Computing at interval'+str(i+1)+'/'+str(len(date_range)))
                 
            logging.info('Filtering users...')
            elites = filter_elites(infile2, threshold, i)
            
            if len(elites) > 1: 
                    
                logging.info('Generating connectivity net...')
                if alpha!=1:
                    connectivity = compute_connectivity(elites, infile1, is_interval, date_range[i+1]-time_delta, date_range[i+1])
                else:
                    connectivity = compute_connectivity(elites, infile1, is_interval, date_range[0], date_range[i+1])
                
                logging.info('Computing phi and alpha')          
                phi, d = compute_phi_d(elites, connectivity)
                
                y = ssd.squareform(d)
                
                # Call clustering.py with the appropriate algorithm and method.
                # clustering.py calculates linkage matrix and polarization 
                logging.info('Computing clusters...')    
                Z, pol = cl.agglomerative_clustering(y, method = method, alpha = 1, K=None, verbose = 0, algorithm=algorithm)
                print(Z)
                print(pol)
                
                logging.info('Plotting dendrogram')  
                plot_dendrogram(elites, Z, pol, date_range[i+1]) 
                
            else:
                logging.info('Not enough elites.')
                
    else:
        logging.info('Filtering users...')
        elites = filter_elites(infile2, threshold, 0)
        
        if len(elites) > 1: 
            connectivity = compute_connectivity(elites, infile1, is_interval, None, None)
                
            logging.info('Computing phi and alpha')          
            phi, d = compute_phi_d(elites, connectivity)
                
            y = ssd.squareform(d)
                
            # Call clustering.py with the appropriate algorithm and method.
            # clustering.py calculates linkage matrix and polarization 
            logging.info('Computing clusters...')    
            Z, pol = cl.agglomerative_clustering(y, method = method, alpha = 1, K=None, verbose = 0, algorithm=algorithm)
            print(Z)
            print(pol)
                
            logging.info('Plotting dendrogram')  
            plot_dendrogram(elites, Z, pol, 'Accumulated') 
                
        else:
            logging.info('Not enough elites.')
            
    logging.info('Finished.')        

    infile1.close()
    infile2.close()    
    
# Determine elites from file created by changes.py and effective threshold.
# Effective threshold is highest between threshold used by changes.py 
# and dendrogram.py
def filter_elites(infile2, threshold, i):
    elites ={} #dictionary with elite usernames and corresponding index
    infile2.seek(0)
    csv_file = csv.reader(infile2)
    
    number_of_line = 0
    index = 0
    for line in csv_file:
        if number_of_line > 0:
            sum_score = float(line[2+i])
            if sum_score >= threshold:
                elites[(str(line[1]))] = index
                index+=1
        number_of_line = number_of_line + 1
        
    

    return elites

# connectivity is an array of vectors the vectors of retweeting users for 
# each corresponding elite. 0 not retweeted, 1 retweeted
def compute_connectivity(elites, infile1, is_interval, start_time, end_time):
    print(is_interval)
    connectivity = np.zeros((len(elites),2), dtype=bool)
    vector0 = np.zeros((len(elites)), dtype=bool)
    retweeters ={}
    index = 0 
    infile1.seek(0)
    
    for line in infile1:
        for tweet in ensure_flattened(json.loads(line)):
            if 'referenced_tweets' in tweet:
                for x in tweet['referenced_tweets']:
                    if 'retweeted' in x['type']:
                        author_name = x['author']['username']
                        retweeter = tweet['author']['username']
                        created_at = tweet['created_at']
                        is_allowed = True
                        if is_interval:
                            created_at_time = pd.to_datetime(created_at, utc=True)
                            if not start_time <= created_at_time <= end_time:
                                is_allowed = False
                        if is_allowed:
                            if author_name in elites:
                                if retweeter not in retweeters:
                                    retweeters[retweeter]= index
                                    if index > 1:
                                        connectivity = np.column_stack((connectivity,vector0))
                                    index+=1
                                connectivity[elites[author_name],retweeters[retweeter]]=1

    return connectivity
    

# Compute phi and d for each pair of elites
def  compute_phi_d(elites, connectivity):
    phi = np.zeros((len(elites),len(elites)))
    d = np.zeros((len(elites),len(elites)))
  
    for i in range(len(elites)):
        for j in range(i):
            f11 = np.sum(connectivity[i,:]*connectivity[j,:])
            f00 = np.sum(np.invert(connectivity[i,:])*np.invert(connectivity[j,:]))
            f10 = np.sum(connectivity[i,:]*np.invert(connectivity[j,:]))
            f01 = np.sum(np.invert(connectivity[i,:])*connectivity[j,:])
            f1x = np.sum(connectivity[i,:])
            f0x = np.sum(np.invert(connectivity[i,:]))
            fx1 = np.sum(connectivity[j,:])
            fx0 = np.sum(np.invert(connectivity[j,:]))          
            
            phi[i,j] = (f11*f00-f10*f01)/(np.sqrt(f1x)*np.sqrt(f0x)*np.sqrt(fx1)*np.sqrt(fx0)) #!!May generate overflow for very big samples 
            phi[j,i] = phi[i,j]
            d[i,j] = np.sqrt(2*(1-phi[i,j]))
            d[j,i]= d[i,j]
            
    return phi, d

# Plot dendrogram
def plot_dendrogram(elites, Z, pol, date):
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 5))
    elite_authors=np.fromiter(elites.keys(),dtype="<U20")
    dendro = sch.dendrogram(Z, labels=elite_authors, ax=ax, orientation='top', distance_sort='ascending', color_threshold = 1*max(Z[:,2]))
    plt.xticks(rotation=45, ha='right')
    plt.savefig('dendrogram_'+str(date)[0:10]+'.png', format='png', bbox_inches='tight', dpi=1200)   

    plt.figure()
    plt.plot(pol)
    plt.savefig('polarisation_'+str(date)[0:10]+'.png', format='png', bbox_inches='tight', dpi=1200)   
    

def set_score_value(username, score, dictionary):
    dictionary[username] = score

if __name__ == '__main__':
    main()