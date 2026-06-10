# -*- coding: utf-8 -*-
"""
Created on Thu May 28 08:34:36 2026

@author: leoai

Description :
    Simulation d'un algorithme génétique appliqué à la théorie des perspectives (Prospect Theory).
    Les agents sont caractérisés par trois paramètres :
        - alpha  : distorsion des probabilités (fonction de pondération de Prelec)
        - beta   : attitude face au risque (courbure de la fonction d'utilité)
        - k      : sensibilité au différentiel d'utilité (paramètre de la sigmoïde de choix)
    L'algorithme fait évoluer la population sur plusieurs générations via sélection,
    reproduction et mutation, dans deux contextes distincts : gains et pertes.
"""

import numpy as np
import random
import matplotlib.pyplot as plt


random.seed(123)
np.random.seed(123)

# ── Paramètres globaux de la simulation ──

#Nombre de générations
epoque=500

#Nombre de tests effectués par chaque agent
nb_test = 100

#Nombre de probas dans chaque loterie
N_L=100
proba = [i/N_L for i in range(1,N_L+1)]

# ── Paramètres de l'algorithme génétique ──
taux_mutation = 0.08    # Part d'agents remplacés aléatoirement à chaque génération
taux_melange = 0.25     # Amplitude maximale du mélange convexe entre deux parents
taux_selection = 0.2    # Part des meilleurs agents conservés pour la reproduction

# Bornes du domaine de recherche pour alpha et beta
alpha_min,alpha_max = 0.25,1.75
beta_min,beta_max = -0.80,0.80

# Tirage aléatoire d'un couple (alpha,beta) — utilisé uniquement pour initialisation
alpha = random.uniform(alpha_min,alpha_max)
beta = random.uniform(beta_min,beta_max)

# Bornes pour le paramètre de sensibilité k (sigmoïde de choix)
k_min,k_max=0.1,10

# ── Fonctions de la théorie des perspectives ──

def ponderation(p,alpha):
    """
    Fonction de pondération de Prelec (1998).
    Transforme une probabilité objective p en probabilité subjective w(p).
    
    Pour alpha < 1 : sur-pondération des petites probabilités et sous-pondération
                     des grandes (forme en S inversé, conforme à la Prospect Theory).
    Pour alpha = 1 : w(p) = p (cas rationnel / espérance d'utilité classique).
    Pour alpha > 1 : sous-pondération des petites probabilités.

    Parameters
    ----------
    p : float
        Probabilité objective, dans [0, 1].
    alpha : float
        Paramètre de distorsion. alpha dans [0.25, 1.75] dans cette simulation.

    Returns
    -------
    float
        Probabilité subjective w(p) ∈ [0, 1].
    """
    if p==0:
        return 0
    else:
        return np.exp(-(-np.log(p))**alpha)


def utilite(x,beta):
    """
    Fonction d'utilité à aversion au risque variable.
    
    - Pour x > 0 (gains)  : u(x) = x^(1-beta). Concave si beta > 0 (aversion au risque),
                             convexe si beta < 0 (goût du risque).
    - Pour x < 0 (pertes) : u(x) = -|x|^(1+beta). Convexe si beta > 0 (aversion aux pertes
                             .
    - Pour x = 0          : u(0) = 0.

    Parameters
    ----------
    x : float
        Montant du gain ou de la perte.
    beta : float
        Paramètre de courbure. beta dans [-0.8, 0.8] dans cette simulation.

    Returns
    -------
    float
        Utilité subjective du montant x.
    """
    if x>0:
        return x**(1-beta)
    elif x<0:
        return -abs(x)**(1+beta)
    else:
        return  0
    
def gain1(): 
    """
    Construit le vecteur des gains associés à chaque probabilité.
    
    Principe : pour chaque probabilité p, le gain est 1/p.
    Ainsi, l'espérance mathématique de chaque loterie est identique (= 1),
    ce qui garantit une comparaison équitable entre loteries.

    Returns
    -------
    gain : list
        Liste des gains correspondant à chaque probabilité dans proba.
    """          
    gain=[]
    for i in proba:
        if i==0:
            gain.append(0)
        else: 
            gain.append(1/i)
    return gain

gain=gain1()
 


def loterie(): 
    """
    Construit la liste complète des loteries disponibles.
    
    Chaque loterie est un couple (probabilité, gain) de la forme (p, 1/p),
    garantissant une espérance mathématique constante égale à 1.

    Returns
    -------
    L : list of tuples
        Liste de loteries sous la forme (p, gain), ou False si les vecteurs
        proba et gain ont des longueurs différentes.
    """         
    if len(proba)!=len(gain):
        return False
    else:
        L=[]
        for i in range(len(proba)):
            l_i = (proba[i],gain[i])
            L.append(l_i)    
    return L

# Construction de l'ensemble des loteries disponibles (gains)
L=loterie()

def echantillon():
    """
    Génère une population initiale de 100 agents tirés aléatoirement.
    
    Chaque agent est un triplet (alpha, beta, k) :
        - alpha : paramètre de distorsion des probabilités, tiré dans [alpha_min, alpha_max]
        - beta  : paramètre de courbure de l'utilité, tiré dans [beta_min, beta_max]
        - k     : sensibilité au différentiel d'utilité (sigmoïde), tiré dans [k_min, k_max]

    Returns
    -------
    population : list of tuples
        Liste de 100 agents, chacun représenté par (alpha, beta, k).
    """         
    population = []
    c=1
    while c<=100:
        alpha = random.uniform(alpha_min,alpha_max)
        beta = random.uniform(beta_min,beta_max)
        k=random.uniform(k_min,k_max)
        population.append((alpha,beta,k))
        c=c+1
    return population
# Population initiale pour le mécanisme de gains
m=echantillon()


def choix_proba(agent,l1,l2):
    """
    Détermine stochastiquement le choix d'un agent entre deux loteries.
    
    L'agent calcule la valeur subjective (SEU) de chaque loterie selon
    la Prospect Theory : V(l) = w(p) * u(x). 
    La probabilité de choisir l1 est donnée par une fonction sigmoïde
    paramétrée par k, qui contrôle la sensibilité au différentiel de valeur :
        P(l1) = 1 / (1 + exp(-(V1-V2)/k))
    Un k faible = choix quasi-déterministe ; un k élevé = choix quasi-aléatoire.

    Parameters
    ----------
    agent : tuple (alpha, beta, k)
        Paramètres comportementaux de l'agent.
    l1 : tuple (p1, x1)
        Première loterie.
    l2 : tuple (p2, x2)
        Deuxième loterie.

    Returns
    -------
    tuple
        La loterie choisie (l1 ou l2).
    """
    a,b,k=agent
    p1,x1=l1
    p2,x2=l2
    # Calcul des valeurs subjectives selon la Prospect Theory
    V1=ponderation(p1,a)*utilite(x1,b)
    V2=ponderation(p2,a)*utilite(x2,b)
    
    
    argument = np.clip(-(V1-V2)/k, -500, 500)
    # Probabilité de choisir l1
    p_choix_l1=1/(1+np.exp(argument))
       
  
    if random.random()<p_choix_l1:
        return l1
    else:
        return l2 
    
def choix_agent():
    """
    Fait choisir chaque agent de la population courante entre deux loteries tirées aléatoirement.

    Returns
    -------
    choix : list of tuples
        Liste des loteries choisies par chaque agent.
    """
    choix=[]                
    for agent in m:
        # Tirage sans remise de deux loteries distinctes pour chaque agent
        l1,l2=random.sample(L,2)
        choix.append(choix_proba(agent,l1,l2))
    return choix

choix=choix_agent()    


########____Représentation graphique____########
# ── Graphiques de la population initiale (avant évolution) ──
fig, axes = plt.subplots(1, 2, figsize=(12,5))
### Fonction pondération
x = np.linspace(0.01,1,100)
def alpha1():
    # Extraction des valeurs alpha de tous les agents
    return [parametre_agent[0] for parametre_agent in m]
parametre1=alpha1()

# Tracé de la fonction de pondération pour chaque agent
for alpha in parametre1:
    y = [ponderation(p, alpha) for p in x]
    axes[0].plot(x, y, color="orange")

# Droite de référence w(p) = p (agent rationnel)
axes[0].plot(x, x, "--", color="black")
axes[0].set_title("Fonction de pondération")
axes[0].set_xlabel("p")
axes[0].set_ylabel("w(p)")
axes[0].set_xlim(0,1)
axes[0].set_ylim(0,1)
axes[0].grid(True)

#######_____GAIN_____#######
x = np.linspace(0,1,100)

def beta2():
    # Extraction des valeurs beta de tous les agents
    return [parametre_agent[1] for parametre_agent in m]
parametre2=beta2()

# Tracé de la fonction d'utilité (gains) pour chaque agent
for beta in parametre2:
    y = [utilite(p, beta) for p in x]
    axes[1].plot(x, y, color="orange")

# Droite de référence u(x) = x (utilité linéaire / neutralité au risque)
axes[1].plot(x, x, "--", color="black")
axes[1].set_title("Fonction d'utilité")
axes[1].set_xlabel("x")
axes[1].set_ylabel("u(x)")
axes[1].set_xlim(0,1)
axes[1].set_ylim(0,1)
axes[1].grid(True)

plt.tight_layout()
plt.show()
### Fonction utility

######_____LOSS_____######
# Tracé de la fonction d'utilité sur le domaine des pertes (x dans [-1, 0])
x = np.linspace(-1,0,100)

def beta1():
    # Extraction des valeurs beta de tous les agents
    return [parametre_agent[1] for parametre_agent in m]
parametre2=beta1()

# Tracé de la fonction d'utilité (pertes) pour chaque agent
for beta in parametre2:
    y = [utilite(p, beta) for p in x]
    plt.plot(x, y,color="orange")
plt.plot(x, x,"--",color="black")
plt.title("Utility function")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.show()


def gain_agent(agent, loterie):
    """
    Simule un tirage pour un agent donné : il choisit une loterie puis joue le hasard.
    
    L'agent sélectionne une loterie parmi deux tirées aléatoirement, puis le résultat
    est réalisé : il reçoit le gain x avec probabilité p, ou 0 sinon.

    Parameters
    ----------
    agent : tuple (alpha, beta, k)
        Paramètres comportementaux de l'agent.
    loterie : list of tuples
        Liste de loteries (probabilité, gain) utilisée pour le tirage.

    Returns
    -------
    float
        Gain réalisé (x si succès, 0 sinon).
    """    
    l1, l2 = random.sample(loterie, 2)
    choix = choix_proba(agent, l1, l2)
    p, x = choix
    # Réalisation aléatoire de la loterie choisie
    if random.random() < p:
        return x
    else:
        return 0

def evaluation_population(population, loterie):
    """
    Évalue chaque agent de la population en cumulant ses gains sur nb_test tirages.
    
    Le score d'un agent est la somme de ses gains réalisés sur nb_test épreuves
    indépendantes. Ce score sert de critère de sélection dans l'algorithme génétique.

    Parameters
    ----------
    population : list of tuples
        Liste des agents à évaluer.
    loterie : list of tuples
        Liste de loteries utilisée pour l'évaluation.

    Returns
    -------
    gain_final : list of tuples (agent, score)
        Liste des agents accompagnés de leur score cumulé.
    """
    gain_final = []
    for agent in population:
        score = 0
        for _ in range(nb_test):
            score += gain_agent(agent, loterie)
        gain_final.append((agent, score))
    return gain_final

def selection(gain_final):
    """
    Sélectionne les meilleurs agents selon leur score cumulé.
    
    Les agents sont triés par score décroissant, et seule la fraction
    supérieure (définie par taux_selection) est conservée pour la reproduction.

    Parameters
    ----------
    gain_final : list of tuples (agent, score)
        Liste des agents avec leur score.

    Returns
    -------
    meilleur : list of tuples
        Liste des agents sélectionnés (les taux_selection * 100 % meilleurs).
    """
    # Tri décroissant par score
    gain_final.sort(key=lambda x: x[1], reverse=True)
    agent_gardes = int(len(gain_final) * taux_selection)
    meilleur = [agent for agent, _ in gain_final[:agent_gardes]]
    return meilleur

def reproduction(meilleur):
    """
    Génère la prochaine génération par interpolation linéaire.
    
    À partir de la liste des meilleurs agents, deux enfants sont produits par
    croisement : pour chaque paire de parents (p1, p2), on tire lambda dans [0, taux_melange]
    et on crée x1 = lambda*p1 + (1-lambda)*p2 et x2 = lambda*p2 + (1-lambda)*p1.
    La génération suivante commence par les parents conservés (élitisme).

    Parameters
    ----------
    meilleur : list of tuples
        Agents sélectionnés à partir desquels se fait la reproduction.

    Returns
    -------
    generation_suivante : list of tuples
        Nouvelle population de 100 agents.
    """

    generation_suivante = meilleur.copy()
    while len(generation_suivante) < 100: 
        valeur_lambda = random.uniform(0, taux_melange)
        parent1 = random.choice(meilleur)
        parent2 = random.choice(meilleur)
        a1, b1, k1 = parent1
        a2, b2, k2 = parent2
        # Création de deux enfants par interpolation linéaire
        x1 =(valeur_lambda * a1 + (1 - valeur_lambda) * a2,valeur_lambda * b1 + (1 - valeur_lambda) * b2,valeur_lambda * k1 + (1 - valeur_lambda) * k2)
        x2 = (valeur_lambda * a2 + (1 - valeur_lambda) * a1,valeur_lambda * b2 + (1 - valeur_lambda) * b1, valeur_lambda * k2 + (1 - valeur_lambda) * k1)
        generation_suivante.append(x1)
        if len(generation_suivante) < 100:
            generation_suivante.append(x2)
    return generation_suivante


def mutation(generation_suivante):
    """
    Introduit de la diversité génétique par remplacement aléatoire de quelques agents.
    
    Une partie (taux_mutation) de la population est remplacée par de nouveaux agents
    aux paramètres tirés aléatoirement.

    Parameters
    ----------
    generation_suivante : list of tuples
        Population après reproduction.

    Returns
    -------
    generation_suivante : list of tuples
        Population après mutation.
    """
    agent_mutes = int(len(generation_suivante) * taux_mutation)
    for _ in range(agent_mutes):
        # Sélection aléatoire d'un agent à remplacer
        nouveau = random.randint(0, len(generation_suivante) - 1)
        alpha = random.uniform(alpha_min, alpha_max)
        beta = random.uniform(beta_min, beta_max)
        k_agent=random.uniform(k_min, k_max)
        generation_suivante[nouveau] = (alpha, beta, k_agent)
    return generation_suivante

def evolution(population, loterie):
    """
    Effectue un cycle complet de l'algorithme génétique sur une génération.
    
    Enchaîne les quatre étapes :
        1. Évaluation : calcul du score cumulé de chaque agent.
        2. Sélection  : conservation des meilleurs agents.
        3. Reproduction : génération de la nouvelle population par croisement.
        4. Mutation   : introduction de diversité aléatoire.

    Parameters
    ----------
    population : list of tuples
        Population courante.
    loterie : list of tuples
        Liste de loteries utilisée pour l'évaluation.

    Returns
    -------
    generation_suivante : list of tuples
        Population de la génération suivante.
    """
    gain_final = evaluation_population(population, loterie)
    meilleur = selection(gain_final)
    generation_suivante = reproduction(meilleur)
    generation_suivante = mutation(generation_suivante)
    return generation_suivante

######## Evolution de alpha, beta et k ########
# ── Stockage de l'historique des paramètres au fil des générations (gains) ──

matrice_alpha = []   
matrice_beta = []   
matrice_k = []       

# ── Boucle principale de l'algorithme génétique — Mécanisme de gains ──
for _ in range(epoque):
    # Une génération complète : évaluation → sélection → reproduction → mutation
    m = evolution(m, L)
    # Enregistrement des paramètres de la génération courante
    alpha_generation = [agent[0] for agent in m]
    beta_generation = [agent[1] for agent in m]
    k_generation = [agent[2] for agent in m]
    matrice_alpha.append(alpha_generation)
    matrice_beta.append(beta_generation)
    matrice_k.append(k_generation)


matrice_alpha = np.array(matrice_alpha).T
matrice_beta = np.array(matrice_beta).T
matrice_k = np.array(matrice_k).T


# ── Heatmap alpha (gains) ──
plt.figure(figsize=(12,6))

plt.imshow(matrice_alpha, aspect='auto', cmap='viridis', interpolation='nearest', origin='upper')
plt.colorbar(label="Valeur de alpha")
plt.xlabel("Génération")
plt.ylabel("Agent")
plt.title("Évolution des paramètres alpha")

plt.tight_layout()
plt.show()


# ── Heatmap beta (gains) ──
plt.figure(figsize=(12,6))

plt.imshow(matrice_beta, aspect='auto', cmap='viridis', interpolation='nearest', origin='upper')
plt.colorbar(label="Valeur de beta")
plt.xlabel("Génération")
plt.ylabel("Agent")
plt.title("Évolution des paramètres beta")

plt.tight_layout()
plt.show()


# ── Heatmap k (gains) ──
plt.figure(figsize=(12,6))

plt.imshow(matrice_k, aspect='auto', cmap='viridis', interpolation='nearest', origin='upper')
plt.colorbar(label="Valeur de k")
plt.xlabel("Génération")
plt.ylabel("Agent")
plt.title("Évolution des paramètres k")
plt.tight_layout()
plt.show()

# ── Graphiques de la population finale après évolution (gains) ──       
########____Représentation graphique____########
fig, axes = plt.subplots(1, 2, figsize=(12,5))
### Fonction pondération
x = np.linspace(0.01,1,100)
def alpha1():
    return [parametre_agent[0] for parametre_agent in m]
parametre1=alpha1()

# Tracé de la fonction de pondération pour chaque agent de la population finale
for alpha in parametre1:
    y = [ponderation(p, alpha) for p in x]
    axes[0].plot(x, y, color="orange")

axes[0].plot(x, x, "--", color="black")
axes[0].set_title("Fonction de pondération")
axes[0].set_xlabel("p", fontsize=13)
axes[0].set_ylabel("w(p)", fontsize=13)
axes[0].set_xlim(0,1)
axes[0].set_ylim(0,1)
axes[0].grid(True)
axes[0].tick_params(labelsize=11)

#######_____GAIN_____#######
x = np.linspace(0,1,100)

def beta2():
    return [parametre_agent[1] for parametre_agent in m]
parametre2=beta2()

# Tracé de la fonction d'utilité (gains) pour chaque agent de la population finale
for beta in parametre2:
    y = [utilite(p, beta) for p in x]
    axes[1].plot(x, y, color="orange")

axes[1].plot(x, x, "--", color="black")
axes[1].set_title("Fonction d'utilité")
axes[1].set_xlabel("x", fontsize=13)
axes[1].set_ylabel("u(x)", fontsize=13)
axes[1].set_xlim(0,1)
axes[1].set_ylim(0,1)
axes[1].grid(True)
axes[1].tick_params(labelsize=11)

plt.tight_layout()
plt.show()



########____SIGMOÏDE DE CHOIX  GAIN____########
# ── Visualisation de l'effet du paramètre k sur la sigmoïde de choix ──
# Affiche une courbe par agent de la population finale

difference_SEU = np.linspace(-20, 20, 500)

def valeur_k():
    # Extraction des valeurs k de la population finale
    return [parametre_agent[2] for parametre_agent in m]
valeur_k1=valeur_k()

plt.figure()

# Une courbe sigmoïde par agent, colorée par défaut (matplotlib)
for k in valeur_k1:
    probabilite = []
    for d in difference_SEU:
        p = 1 / (1 + np.exp(-d / k))
        probabilite.append(p)
    plt.plot(difference_SEU, probabilite)

plt.title("Effet du paramètre k sur la probabilité de choix")
plt.xlabel("SEU(L1) - SEU(L2)")
plt.ylabel("P(choix L1)")
plt.grid(True)
plt.show()


# ── Comparaison des sigmoïdes pour des valeurs fixes de k ──
# Illustration pédagogique de l'effet du paramètre k

difference_SEU = np.linspace(-20, 20, 500)
valeurs_k = [0.1,0.5, 1, 2, 5, 10]
plt.figure(figsize=(10,6))
for k in valeurs_k:
    probabilite = []
    for d in difference_SEU:
        p = 1 / (1 + np.exp(-d / k))
        probabilite.append(p)
    plt.plot(difference_SEU, probabilite, label=f"k = {k}")

# Mise en forme
plt.title("Effet du paramètre k sur la fonction sigmoïde")
plt.xlabel("V1 - V2")
plt.ylabel("Probabilité de choisir L1")
plt.xlim(-20,20)
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.show()


# ── Évolution de la moyenne de alpha au fil des générations (gains) ──
mean_alpha = np.mean(matrice_alpha, axis=0)
plt.figure(figsize=(10,5))
plt.plot(mean_alpha)
plt.xlabel("Génération")
plt.ylabel("Alpha moyen")
plt.title("Évolution moyenne du paramètre alpha")
plt.grid(True)
plt.show()


# ── Évolution de la moyenne de beta au fil des générations (gains) ──
mean_beta = np.mean(matrice_beta, axis=0)
plt.figure(figsize=(10,5))
plt.plot(mean_beta)
plt.xlabel("Génération")
plt.ylabel("Beta moyen")
plt.title("Évolution moyenne du paramètre beta")
plt.grid(True)
plt.show()


# ── Évolution de la moyenne de k au fil des générations (gains) ──
mean_k= np.mean(matrice_k, axis=0)
plt.figure(figsize=(10,5))
plt.plot(mean_k)
plt.xlabel("Génération")
plt.ylabel("k moyen")
plt.title("Evolution moyenne du paramètre k")
plt.grid(True)
plt.show()


# ── Figures combinées (courbe moyenne + heatmap) pour alpha — Gains ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
moyenne=np.mean(mean_alpha)
sigma=np.std(mean_alpha)
ax1.plot(mean_alpha, color='blue', lw=2)
# Annotation de la moyenne et de l'écart-type sur le graphique
ax1.text(
    0.02, 0.95,
    fr'$\alpha = {moyenne:.3f} \pm {sigma:.3f}$',
    transform=ax1.transAxes,
    fontsize=12,
    verticalalignment='top'
)
ax1.set_title("Evolution moyenne du paramètre alpha")
ax1.set_xlabel("Génération")
ax1.set_ylabel("Alpha moyen")
ax1.grid(True)
im = ax2.imshow(matrice_alpha, aspect='auto', cmap='viridis', origin='lower')
ax2.set_title("Evolution des paramètres alpha")
ax2.set_xlabel("Génération")
ax2.set_ylabel("Indice du paramètre")
fig.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()


# ── Figures combinées (courbe moyenne + heatmap) pour beta — Gains ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
moyenne=np.mean(mean_beta)
sigma=np.std(mean_beta)
ax1.plot(mean_beta, color='blue', lw=2)
ax1.text(0.02, 0.95, fr'$\beta = {moyenne:.3f} \pm {sigma:.3f}$', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
ax1.set_title("Evolution moyenne du paramètre beta")
ax1.set_xlabel("Génération")
ax1.set_ylabel("Beta moyen")
ax1.grid(True)
im = ax2.imshow(matrice_beta, aspect='auto', cmap='viridis', origin='lower')
ax2.set_title("Evolution des paramètres beta")
ax2.set_xlabel("Génération")
ax2.set_ylabel("Indice du paramètre")
fig.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()


# ── Figures combinées (courbe moyenne + heatmap) pour k — Gains ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
moyenne=np.mean(mean_k)
sigma=np.std(mean_k)
ax1.plot(mean_k, color='blue', lw=2)
ax1.text(0.02, 0.95, fr'k = {moyenne:.3f} \pm {sigma:.3f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
ax1.set_title("Evolution moyenne du paramètre k")
ax1.set_xlabel("Génération")
ax1.set_ylabel("K moyen")
ax1.grid(True)
im = ax2.imshow(matrice_k, aspect='auto', cmap='viridis', origin='lower')
ax2.set_title("Evolution des paramètres k")
ax2.set_xlabel("Génération")
ax2.set_ylabel("Indice du paramètre")
fig.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()


###############################################################
#                                                             #
#                    MÉCANISME DE PERTES                      #
#                                                             #
###############################################################

perte = [-1/p for p in proba]

def loterie_perte():
    """
    Construit la liste des loteries de perte.
    
    Chaque loterie est un couple (probabilité, perte) de la forme (p, -1/p),
    garantissant une espérance mathématique constante égale à -1.

    Returns
    -------
    L_perte : list of tuples
        Liste de loteries de perte sous la forme (p, -1/p).
    """
    L_perte = []
    for i in range(len(proba)):
        l_i = (proba[i], perte[i])
        L_perte.append(l_i)
    return L_perte

# Construction de l'ensemble des loteries disponibles (pertes)
L_perte = loterie_perte()

# ── Population initiale indépendante pour les pertes ──
m_perte = echantillon()

# ── Représentation graphique population initiale pertes ──
######_____LOSS initiale_____######
# Tracé de la fonction d'utilité sur le domaine des pertes pour la population initiale
x = np.linspace(-1, 0, 100)
beta_perte_init = [agent[1] for agent in m_perte]

for beta in beta_perte_init:
    y = [utilite(v, beta) for v in x]
    plt.plot(x, y, color="blue")
plt.plot(x, x, "--", color="black")
plt.title("Fonction d'utilité — Pertes (population initiale)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.show()

# ── Stockage de l'historique des paramètres au fil des générations (pertes) ──
matrice_alpha_perte = []
matrice_beta_perte  = []
matrice_k_perte     = []

# ── Boucle principale de l'algorithme génétique — Mécanisme de pertes ──
for _ in range(epoque):
    # Une génération complète sur les loteries de perte
    m_perte = evolution(m_perte, L_perte)
    # Enregistrement des paramètres de la génération courante
    alpha_generation_perte = [agent[0] for agent in m_perte]
    beta_generation_perte  = [agent[1] for agent in m_perte]
    k_generation_perte     = [agent[2] for agent in m_perte]
    matrice_alpha_perte.append(alpha_generation_perte)
    matrice_beta_perte.append(beta_generation_perte)
    matrice_k_perte.append(k_generation_perte)


matrice_alpha_perte = np.array(matrice_alpha_perte).T
matrice_beta_perte  = np.array(matrice_beta_perte).T
matrice_k_perte     = np.array(matrice_k_perte).T

# ── Heatmaps pertes ──
plt.figure(figsize=(12, 6))
plt.imshow(matrice_alpha_perte, aspect='auto', cmap='viridis', interpolation='nearest', origin='upper')
plt.colorbar(label="Valeur de alpha")
plt.xlabel("Génération")
plt.ylabel("Agent")
plt.title("Évolution des paramètres alpha — Pertes")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(matrice_beta_perte, aspect='auto', cmap='viridis', interpolation='nearest', origin='upper')
plt.colorbar(label="Valeur de beta")
plt.xlabel("Génération")
plt.ylabel("Agent")
plt.title("Évolution des paramètres beta — Pertes")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(matrice_k_perte, aspect='auto', cmap='viridis', interpolation='nearest', origin='upper')
plt.colorbar(label="Valeur de k")
plt.xlabel("Génération")
plt.ylabel("Agent")
plt.title("Évolution des paramètres k — Pertes")
plt.tight_layout()
plt.show()

# ── Représentation graphique population finale pertes ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Fonction de pondération — population finale pertes
x = np.linspace(0.01, 1, 100)
alpha_perte_final = [agent[0] for agent in m_perte]
for alpha in alpha_perte_final:
    y = [ponderation(p, alpha) for p in x]
    axes[0].plot(x, y, color="blue")
axes[0].plot(x, x, "--", color="black")
axes[0].set_title("Fonction de pondération — Pertes")
axes[0].set_xlabel("p")
axes[0].set_ylabel("w(p)")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].grid(True)

# Fonction d'utilité (pertes) — population finale pertes
x = np.linspace(-1, 0, 100)
beta_perte_final = [agent[1] for agent in m_perte]
for beta in beta_perte_final:
    y = [utilite(v, beta) for v in x]
    axes[1].plot(x, y, color="blue")
axes[1].plot(x, x, "--", color="black")
axes[1].set_title("Fonction d'utilité — Pertes")
axes[1].set_xlabel("x")
axes[1].set_ylabel("u(x)")
axes[1].set_xlim(-1, 0)
axes[1].set_ylim(-1, 0)
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ── Évolution moyenne alpha, beta, k — Pertes ──
mean_alpha_perte = np.mean(matrice_alpha_perte, axis=0)
plt.figure(figsize=(10, 5))
plt.plot(mean_alpha_perte, color="blue")
plt.xlabel("Génération")
plt.ylabel("Alpha moyen")
plt.title("Évolution moyenne du paramètre alpha — Pertes")
plt.grid(True)
plt.show()

mean_beta_perte = np.mean(matrice_beta_perte, axis=0)
plt.figure(figsize=(10, 5))
plt.plot(mean_beta_perte, color="blue")
plt.xlabel("Génération")
plt.ylabel("Beta moyen")
plt.title("Évolution moyenne du paramètre beta — Pertes")
plt.grid(True)
plt.show()

mean_k_perte = np.mean(matrice_k_perte, axis=0)
plt.figure(figsize=(10, 5))
plt.plot(mean_k_perte, color="blue")
plt.xlabel("Génération")
plt.ylabel("k moyen")
plt.title("Évolution moyenne du paramètre k — Pertes")
plt.grid(True)
plt.show()

# ── Figures combinées (courbe moyenne + heatmap) pour alpha — Pertes ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
moyenne = np.mean(mean_alpha_perte)
sigma   = np.std(mean_alpha_perte)
ax1.plot(mean_alpha_perte, color='blue', lw=2)
ax1.text(0.02, 0.95, fr'$\alpha = {moyenne:.3f} \pm {sigma:.3f}$', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
ax1.set_title("Evolution moyenne du paramètre alpha — Pertes")
ax1.set_xlabel("Génération")
ax1.set_ylabel("Alpha moyen")
ax1.grid(True)
im = ax2.imshow(matrice_alpha_perte, aspect='auto', cmap='viridis', origin='lower')
ax2.set_title("Evolution des paramètres alpha — Pertes")
ax2.set_xlabel("Génération")
ax2.set_ylabel("Indice du paramètre")
fig.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()

# ── Figures combinées (courbe moyenne + heatmap) pour beta — Pertes ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
moyenne = np.mean(mean_beta_perte)
sigma   = np.std(mean_beta_perte)
ax1.plot(mean_beta_perte, color='blue', lw=2)
ax1.text(0.02, 0.95, fr'$\beta = {moyenne:.3f} \pm {sigma:.3f}$', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
ax1.set_title("Evolution moyenne du paramètre beta — Pertes")
ax1.set_xlabel("Génération")
ax1.set_ylabel("Beta moyen")
ax1.grid(True)
im = ax2.imshow(matrice_beta_perte, aspect='auto', cmap='viridis', origin='lower')
ax2.set_title("Evolution des paramètres beta — Pertes")
ax2.set_xlabel("Génération")
ax2.set_ylabel("Indice du paramètre")
fig.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()

# ── Figures combinées (courbe moyenne + heatmap) pour k — Pertes ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
moyenne = np.mean(mean_k_perte)
sigma   = np.std(mean_k_perte)
ax1.plot(mean_k_perte, color='blue', lw=2)
ax1.text(0.02, 0.95, fr'$k = {moyenne:.3f} \pm {sigma:.3f}$', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
ax1.set_title("Evolution moyenne du paramètre k — Pertes")
ax1.set_xlabel("Génération")
ax1.set_ylabel("K moyen")
ax1.grid(True)
im = ax2.imshow(matrice_k_perte, aspect='auto', cmap='viridis', origin='lower')
ax2.set_title("Evolution des paramètres k — Pertes")
ax2.set_xlabel("Génération")
ax2.set_ylabel("Indice du paramètre")
fig.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()

# ── Sigmoïde de choix — population finale pertes ──
# Affiche une courbe par agent de la population finale pertes
difference_SEU = np.linspace(-20, 20, 500)
k_perte_final = [agent[2] for agent in m_perte]

plt.figure()
for k in k_perte_final:
    probabilite = [1 / (1 + np.exp(-d / k)) for d in difference_SEU]
    plt.plot(difference_SEU, probabilite, color="blue")
plt.title("Effet du paramètre k sur la probabilité de choix — Pertes")
plt.xlabel("SEU(L1) - SEU(L2)")
plt.ylabel("P(choix L1)")
plt.grid(True)
plt.show()