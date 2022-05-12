"""
TD NOTE D'ANALYSE NUMERIQUE GENIE MATHEMATIQUE ALTERNANCE

Par : Loann Boudinot, Tangi Floch et Nicolas Mallegol
"""
import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from PIL import Image

if __name__ == '__main__':

    st.title('TP noté Analyse Numérique du 13/05/2022')
    introduction = """
        Travail réalisé par Loann Boudinot, Tangi Floch et Nicolas Mallegol.
        """
    st.write(introduction)

    # Ecriture de la sidebar
    title_sidebar = st.sidebar.title("PARAMETRE(S)")
    k = st.sidebar.slider("Choisissez votre k (pas de 5)", min_value = 0, max_value = 500, step= 5)


    file = st.file_uploader('Choississez votre image')

    if file:  # si vous avez choisi une image

        def PSNR(tab1, k):
            U, S, V = svd(tab1)
            b = 0
            for i in range(k, np.shape(S)[0]):
                b += S[i] ** 2
            a = (b) / (np.shape(tab1)[0] * np.shape(tab1)[1])
            return 10 * math.log10(255 ** 2 / a)


        def extraireTab(Img, k):
            tab = np.zeros((int(np.shape(Img)[0]), int(np.shape(Img)[1])), int)
            tab = np.copy(Img[:, :, k])
            return tab


        def compression(tab, k):
            U, S, V = svd(tab)
            S1 = np.zeros((np.shape(U)[1], np.shape(V)[0]), float)
            S2 = np.zeros((np.shape(S)[0]), float)
            S2[:k] = S[:k]
            np.fill_diagonal(S1, S2)
            return np.matmul(np.matmul(U, S1), V)


        def reconstitution(tab1, tab2, tab3):
            tab4 = np.zeros((np.shape(tab1)[0], np.shape(tab1)[1], 3), int)
            tab4[:, :, 0] = np.copy(tab1)
            tab4[:, :, 1] = np.copy(tab2)
            tab4[:, :, 2] = np.copy(tab3)
            return tab4


        # Récupération de l'image et affichage de la taille

        img = plt.imread(file)
        st.write("Les dimensions de l'image sont :", np.shape(img)[0], "x", np.shape(img)[1])
        st.write("Le chargement peut parfois etre long selon la taille de l'image, veuillez patienter... (max : 3-4 min)")
        # Récupération des différents canaux

        tab1 = extraireTab(img, 0)
        tab2 = extraireTab(img, 1)
        tab3 = extraireTab(img, 2)

        # Calcul de l'approximation de rang faible pour chaque canaux

        tab5 = compression(tab1, k)
        tab6 = compression(tab2, k)
        tab7 = compression(tab3, k)

        # Calcul du PSNR pour le k choisi dans la sidebar

        err = (PSNR(tab1, k) + PSNR(tab2, k) + PSNR(tab3, k)) / 3

        # Calcul pour tous les k
        err_n = np.zeros((25), float)
        for i in range(25):
            err_n[i] = (PSNR(tab1, 20 * (i + 1)) + PSNR(tab2, 20 * (i + 1)) + PSNR(tab3, 20 * (i + 1))) / 3

        # Affichage de la courbe du PSNR en fonction de k
        st.set_option('deprecation.showPyplotGlobalUse', False) # Pour ne pas afficher ce warning

        st.title("Representation graphique du PSNR moyen en fonction de k")
        plt.plot(range(0, 500, 20), err_n)
        plt.ylabel("PSNR")
        plt.xlabel("Valeurs de k")
        st.pyplot()

        # Affichage du PSNR en fonction de k

        st.write(f"La valeur du PSNR est de {round(err,3)} dB pour k = {k} .")

        # Reconstitution et affichage de l'image compressée
        tab4 = reconstitution(tab5, tab6, tab7)

        st.title("IMAGE ORIGINALE")
        fig_img = plt.figure()
        plt.imshow(img)
        st.pyplot(fig_img)

        st.title("Rouge")
        fig_img = plt.figure()
        plt.imshow(tab1)
        st.pyplot(fig_img)

        st.title("Vert")
        fig_img = plt.figure()
        plt.imshow(tab2)
        st.pyplot(fig_img)

        st.title("Bleu")
        fig_img = plt.figure()
        plt.imshow(tab3)
        st.pyplot(fig_img)

        st.title(f"IMAGE COMPRESSEE AVEC K = {k}")
        fig_tab4 = plt.figure()
        plt.imshow(tab4)
        st.pyplot(fig_tab4)
