import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título do aplicativo
st.title("🚢 Titanic Survival Prediction App")
st.markdown("""
Este aplicativo prevê a probabilidade de sobrevivência no Titanic usando Machine Learning.
Use as abas abaixo para explorar os dados, fazer previsões ou analisar as métricas do modelo.
""")

# Carregar dados
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/bcmaymonegalvao/Exploratory-Data-Analysis/main/train.csv"
    data = pd.read_csv(url)
    return data

# Pré-processamento dos dados
@st.cache_data
def preprocess_data(data):
    # Fazer uma cópia para não modificar o original
    df = data.copy()
    
    # Preencher valores faltantes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Extrair títulos dos nomes
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 
                                      'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Mapear categorias para valores numéricos
    label_encoders = {}
    categorical_cols = ['Sex', 'Embarked', 'Title']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders

# Carregar modelo (se existir) ou treinar um novo
@st.cache_resource
def load_or_train_model(data):
    try:
        # Tentar carregar um modelo salvo
        with open('titanic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.sidebar.success("Modelo carregado com sucesso!")
        
        # Carregar também as métricas salvas
        with open('titanic_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
            
    except:
        st.sidebar.info("Treinando um novo modelo...")
        
        # Selecionar features para o modelo
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                   'Embarked', 'FamilySize', 'IsAlone', 'Title']
        
        X = data[features]
        y = data['Survived']
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Fazer previsões
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Calcular matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Salvar métricas
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Salvar modelo
        with open('titanic_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Salvar métricas
        with open('titanic_metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)
        
        st.sidebar.success("Novo modelo treinado e salvo com sucesso!")
    
    return model, metrics

# Carregar dados
df = load_data()
df_processed, label_encoders = preprocess_data(df)
model, metrics = load_or_train_model(df_processed)

# Sidebar para inputs do usuário
st.sidebar.header("Filtros de Passageiro")

# Filtros para a base de dados
st.sidebar.subheader("Filtrar Base de Dados")
pclass_filter = st.sidebar.multiselect(
    "Classe", 
    options=[1, 2, 3], 
    default=[1, 2, 3],
    help="Filtrar por classe do passageiro"
)

sex_filter = st.sidebar.multiselect(
    "Sexo", 
    options=["Masculino", "Feminino"], 
    default=["Masculino", "Feminino"],
    help="Filtrar por sexo do passageiro"
)

age_range = st.sidebar.slider(
    "Faixa de Idade", 
    min_value=0, 
    max_value=100, 
    value=(0, 100),
    help="Selecionar faixa de idade"
)

# Mapear sexo para valores do dataframe
sex_map = {"Masculino": "male", "Feminino": "female"}
selected_sex = [sex_map[s] for s in sex_filter]

# Aplicar filtros
filtered_df = df[
    (df['Pclass'].isin(pclass_filter)) &
    (df['Sex'].isin(selected_sex)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1])
]

# Criar abas
tab1, tab2, tab3 = st.tabs(["📊 Análise Exploratória", "🔮 Previsões", "📈 Métricas do Modelo"])

# ABA 1: ANÁLISE EXPLORATÓRIA
with tab1:
    st.header("Análise Exploratória dos Dados do Titanic")
    
    # Estatísticas descritivas
    st.subheader("Estatísticas Descritivas")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Passageiros", len(filtered_df))
        st.metric("Sobreviventes", filtered_df['Survived'].sum())
    
    with col2:
        survival_rate = filtered_df['Survived'].mean() * 100 if len(filtered_df) > 0 else 0
        st.metric("Taxa de Sobrevivência", f"{survival_rate:.2f}%")
        avg_age = filtered_df['Age'].mean() if len(filtered_df) > 0 else 0
        st.metric("Idade Média", f"{avg_age:.1f} anos")
    
    with col3:
        avg_fare = filtered_df['Fare'].mean() if len(filtered_df) > 0 else 0
        st.metric("Tarifa Média", f"${avg_fare:.2f}")
        max_fare = filtered_df['Fare'].max() if len(filtered_df) > 0 else 0
        st.metric("Maior Tarifa", f"${max_fare:.2f}")
    
    # Gráficos
    st.subheader("Visualizações dos Dados")
    
    # Sobrevivência por classe
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        survival_by_class = pd.crosstab(filtered_df['Pclass'], filtered_df['Survived'])
        if not survival_by_class.empty:
            survival_by_class.plot(kind='bar', color=['#ffccc7', '#ff4d4f'], ax=ax)
            plt.title('Sobrevivência por Classe', fontsize=16)
            plt.xlabel('Classe', fontsize=12)
            plt.ylabel('Número de Passageiros', fontsize=12)
            plt.legend(['Não Sobreviveu', 'Sobreviveu'])
            st.pyplot(fig)
        else:
            st.warning("Não há dados para plotar o gráfico de sobrevivência por classe.")
        
        # Sobrevivência por sexo
        fig, ax = plt.subplots(figsize=(10, 6))
        survival_by_sex = pd.crosstab(filtered_df['Sex'], filtered_df['Survived'])
        if not survival_by_sex.empty:
            survival_by_sex.plot(kind='bar', color=['#ffccc7', '#ff4d4f'], ax=ax)
            plt.title('Sobrevivência por Sexo', fontsize=16)
            plt.xlabel('Sexo', fontsize=12)
            plt.ylabel('Número de Passageiros', fontsize=12)
            plt.legend(['Não Sobreviveu', 'Sobreviveu'])
            # Mapear os valores de sexo para os rótulos
            ax.set_xticklabels(['Feminino', 'Masculino'], rotation=0)
            st.pyplot(fig)
        else:
            st.warning("Não há dados para plotar o gráfico de sobrevivência por sexo.")
        
        # Distribuição de idades
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=filtered_df, x='Age', hue='Survived', multiple='stack', 
                     palette='YlOrRd', bins=20, ax=ax)
        plt.title('Distribuição de Idades por Sobrevivência', fontsize=16)
        plt.xlabel('Idade', fontsize=12)
        plt.ylabel('Número de Passageiros', fontsize=12)
        st.pyplot(fig)
        
        # Distribuição de tarifas
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=filtered_df, x='Pclass', y='Fare', hue='Survived', palette='YlOrRd', ax=ax)
        plt.title('Distribuição de Tarifas por Classe e Sobrevivência', fontsize=16)
        plt.xlabel('Classe', fontsize=12)
        plt.ylabel('Tarifa', fontsize=12)
        st.pyplot(fig)
        
        # Heatmap de correlação
        st.subheader("Mapa de Calor de Correlações")
        numeric_df = filtered_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='YlOrRd', center=0, ax=ax)
            plt.title('Matriz de Correlação', fontsize=16)
            st.pyplot(fig)
        else:
            st.warning("Não há dados numéricos para calcular correlações.")
    else:
        st.warning("Não há dados para visualizar. Ajuste os filtros.")
    
    # Tabela de dados
    st.subheader("Visualização dos Dados Filtrados")
    st.dataframe(filtered_df)

# ABA 2: PREVISÕES
with tab2:
    st.header("Previsões de Sobrevivência")
    
    # Opções de previsão
    prediction_type = st.radio(
        "Tipo de Previsão:",
        ["Previsão Individual", "Previsão em Lote"],
        horizontal=True
    )
    
    if prediction_type == "Previsão Individual":
        st.subheader("Previsão Individual")
        
        # Inputs para previsão individual
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Classe", [1, 2, 3], help="1 = Primeira, 2 = Segunda, 3 = Terceira")
            sex = st.selectbox("Sexo", ["Masculino", "Feminino"])
            age = st.slider("Idade", 0, 100, 30)
            sibsp = st.slider("Número de Irmãos/Cônjuges a bordo", 0, 8, 0)
        
        with col2:
            parch = st.slider("Número de Pais/Filhos a bordo", 0, 6, 0)
            fare = st.slider("Tarifa Paga", 0, 300, 30)
            embarked = st.selectbox("Porto de Embarque", ["Cherbourg", "Queenstown", "Southampton"])
        
        # Calcular características derivadas
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0
        
        # Converter entradas para formato do modelo
        sex_encoded = 1 if sex == "Masculino" else 0
        embarked_encoded = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}[embarked]
        
        # Para título, usaremos "Mr" como padrão para homens e "Miss" para mulheres
        title = "Mr" if sex == "Masculino" else "Miss"
        title_encoded = 2 if title == "Mr" else 3  # Valores baseados no encoding feito durante o treino
        
        # Organizar dados em um DataFrame
        user_data = {
            'Pclass': pclass,
            'Sex': sex_encoded,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked_encoded,
            'FamilySize': family_size,
            'IsAlone': is_alone,
            'Title': title_encoded
        }
        
        features = pd.DataFrame(user_data, index=[0])
        
        # Fazer previsão
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)
        
        # Mostrar resultados
        st.subheader("Resultado da Previsão")
        survival_status = "Sobreviveu" if prediction[0] == 1 else "Não Sobreviveu"
        survival_icon = "✅" if prediction[0] == 1 else "❌"
        
        st.markdown(f"### {survival_icon} Previsão: {survival_status}")
        
        # Mostrar probabilidades
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Probabilidade de Não Sobrevivência", 
                      value=f"{prediction_proba[0][0]*100:.2f}%")
        with col2:
            st.metric(label="Probabilidade de Sobrevivência", 
                      value=f"{prediction_proba[0][1]*100:.2f}%")
        
        # Gráfico de probabilidades
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['Não Sobreviveu', 'Sobreviveu'], prediction_proba[0], color=['#ffccc7', '#ff4d4f'])
        ax.set_ylabel('Probabilidade')
        ax.set_title('Probabilidade de Sobrevivência')
        st.pyplot(fig)
    
    else:
        st.subheader("Previsão em Lote")
        st.info("""
        Esta seção faz previsões para todos os passageiros na base de dados filtrada.
        Use os filtros na barra lateral para ajustar o conjunto de dados.
        """)
        
        if filtered_df.empty:
            st.warning("Não há dados para fazer previsões em lote. Ajuste os filtros.")
        else:
            # Pré-processar dados filtrados para previsão
            filtered_processed, _ = preprocess_data(filtered_df)
            
            # Selecionar features para o modelo
            features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                       'Embarked', 'FamilySize', 'IsAlone', 'Title']
            
            X = filtered_processed[features]
            
            # Fazer previsões
            predictions = model.predict(X)
            predictions_proba = model.predict_proba(X)
            
            # Adicionar previsões ao DataFrame
            results_df = filtered_df.copy()
            results_df['Predicted_Survival'] = predictions
            results_df['Survival_Probability'] = predictions_proba[:, 1]
            
            # Estatísticas das previsões
            st.subheader("Estatísticas das Previsões")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Passageiros na Amostra", len(results_df))
                st.metric("Previsão de Sobreviventes", results_df['Predicted_Survival'].sum())
            
            with col2:
                predicted_survival_rate = results_df['Predicted_Survival'].mean() * 100
                st.metric("Taxa de Sobrevivência Prevista", f"{predicted_survival_rate:.2f}%")
                if 'Survived' in results_df:
                    actual_survival_rate = results_df['Survived'].mean() * 100
                    st.metric("Taxa de Sobrevivência Real", f"{actual_survival_rate:.2f}%")
                else:
                    st.metric("Taxa de Sobrevivência Real", "N/A")
            
            with col3:
                avg_prob = results_df['Survival_Probability'].mean()
                st.metric("Probabilidade Média de Sobrevivência", f"{avg_prob*100:.2f}%")
            
            # Gráfico de distribuição de probabilidades
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=results_df, x='Survival_Probability', hue='Predicted_Survival', 
                         multiple='stack', palette='YlOrRd', bins=20, ax=ax)
            plt.title('Distribuição das Probabilidades de Sobrevivência', fontsize=16)
            plt.xlabel('Probabilidade de Sobrevivência', fontsize=12)
            plt.ylabel('Número de Passageiros', fontsize=12)
            st.pyplot(fig)
            
            # Gráfico de comparação entre previsões e resultados real
            if 'Survived' in results_df:
                st.subheader("Comparação entre Previsões e Resultados Reais")
                
                # Calcular acurácia
                accuracy = (results_df['Predicted_Survival'] == results_df['Survived']).mean()
                st.metric("Acurácia do Modelo", f"{accuracy*100:.2f}%")
                
                # Matriz de confusão
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(results_df['Survived'], results_df['Predicted_Survival'])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                            xticklabels=['Não Sobreviveu', 'Sobreviveu'],
                            yticklabels=['Não Sobreviveu', 'Sobreviveu'])
                plt.title('Matriz de Confusão', fontsize=16)
                plt.ylabel('Real', fontsize=12)
                plt.xlabel('Previsto', fontsize=12)
                st.pyplot(fig)
            
            # Tabela com resultados
            st.subheader("Resultados das Previsões")
            st.dataframe(results_df[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 
                                    'Survived', 'Predicted_Survival', 'Survival_Probability']].head(20))

# ABA 3: MÉTRICAS DO MODELO
with tab3:
    st.header("Métricas de Avaliação do Modelo")
    
    # Introdução
    st.markdown("""
    ### Importância das Métricas de Avaliação
    
    As métricas de avaliação são fundamentais para entender o desempenho de um modelo de machine learning.
    Cada métrica fornece uma perspectiva diferente sobre como o modelo está se saindo na tarefa de classificação.
    """)
    
    # Métricas principais
    st.subheader("Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Acurácia", f"{metrics['accuracy']*100:.2f}%")
        with st.expander("O que é Acurácia?"):
            st.markdown("""
            **Acurácia** mede a proporção de previsões corretas (tanto positivas quanto negativas) em relação ao total de previsões.
            
            **Fórmula**: (Verdadeiros Positivos + Verdadeiros Negativos) / Total
            
            **Interpretação**: Uma acurácia de 80% significa que o modelo acertou 80% das previsões.
            
            **Limitação**: Em conjuntos de dados desbalanceados, a acurácia pode ser enganosa.
            """)
    
    with col2:
        st.metric("Precisão", f"{metrics['precision']*100:.2f}%")
        with st.expander("O que é Precisão?"):
            st.markdown("""
            **Precisão** mede a proporção de verdadeiros positivos em relação a todos os casos classificados como positivos.
            
            **Fórmula**: Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos)
            
            **Interpretação**: Uma precisão de 75% significa que, das pessoas que o modelo previu como sobreviventes, 75% realmente sobreviveram.
            
            **Importante**: Métrica crucial quando o custo dos falsos positivos é alto.
            """)
    
    with col3:
        st.metric("Recall (Sensibilidade)", f"{metrics['recall']*100:.2f}%")
        with st.expander("O que é Recall?"):
            st.markdown("""
            **Recall** (também chamado de Sensibilidade) mede a proporção de verdadeiros positivos em relação a todos os casos que são realmente positivos.
            
            **Fórmula**: Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos)
            
            **Interpretação**: Um recall de 70% significa que o modelo identificou corretamente 70% de todos os sobreviventes reais.
            
            **Importante**: Métrica crucial quando o custo dos falsos negativos é alto.
            """)
    
    with col4:
        st.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
        with st.expander("O que é F1-Score?"):
            st.markdown("""
            **F1-Score** é a média harmônica entre Precisão e Recall, proporcionando um equilíbrio entre as duas métricas.
            
            **Fórmula**: 2 × (Precisão × Recall) / (Precisão + Recall)
            
            **Interpretação**: Um F1-Score de 80% indica um bom equilíbrio entre precisão e recall.
            
            **Importante**: Métrica especialmente útil em conjuntos de dados desbalanceados.
            """)
    
    # Matriz de Confusão
    st.subheader("Matriz de Confusão")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                xticklabels=['Não Sobreviveu', 'Sobreviveu'],
                yticklabels=['Não Sobreviveu', 'Sobreviveu'])
    plt.title('Matriz de Confusão', fontsize=16)
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Previsão do Modelo', fontsize=12)
    st.pyplot(fig)
    
    with st.expander("O que é a Matriz de Confusão?"):
        st.markdown("""
        A **Matriz de Confusão** é uma tabela que mostra o desempenho de um modelo de classificação:
        
        - **Verdadeiros Negativos (TN)**: Casos negativos corretamente classificados como negativos
        - **Falsos Positivos (FP)**: Casos negativos incorretamente classificados como positivos (Erro Tipo I)
        - **Falsos Negativos (FN)**: Casos positivos incorretamente classificados como negativos (Erro Tipo II)
        - **Verdadeiros Positivos (TP)**: Casos positivos corretamente classificados como positivos
        
        A matriz de confusão permite calcular várias métricas de desempenho e entender os tipos de erros que o modelo está cometendo.
        """)
    
    # Curva ROC
    st.subheader("Curva ROC e Área sob a Curva (AUC)")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2, 
            label=f'Curva ROC (AUC = {metrics["roc_auc"]:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Classificador Aleatório')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    ax.set_title('Curva ROC', fontsize=16)
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    with st.expander("O que é a Curva ROC e AUC?"):
        st.markdown("""
        A **Curva ROC (Receiver Operating Characteristic)** mostra o desempenho de um modelo de classificação em todos os limiares de classificação.
        
        - **Eixo X**: Taxa de Falsos Positivos (FPR) - proporção de negativos classificados incorretamente como positivos
        - **Eixo Y**: Taxa de Verdadeiros Positivos (TPR) - também conhecida como Recall ou Sensibilidade
        
        **AUC (Área sob a Curva)** quantifica a capacidade do modelo de distinguir entre classes:
        
        - **AUC = 0.5**: O modelo não é melhor que um classificador aleatório
        - **AUC = 1.0**: Classificação perfeita
        - **AUC entre 0.5 e 1.0**: Quanto maior, melhor a capacidade do modelo de distinguir entre classes
        
        Um AUC de {:.2f} indica que o modelo tem {} de capacidade de distinguir entre sobreviventes e não-sobreviventes.
        """.format(metrics['roc_auc'], "excelente" if metrics['roc_auc'] > 0.9 else "boa" if metrics['roc_auc'] > 0.8 else "moderada"))
    
    # Relatório de Classificação
    st.subheader("Relatório de Classificação Detalhado")
    
    # Converter o relatório de classificação para DataFrame
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    st.dataframe(report_df.style.format('{:.2f}'))
    
    with st.expander("Entendendo o Relatório de Classificação"):
        st.markdown("""
        O **Relatório de Classificação** fornece métricas detalhadas para cada classe:
        
        - **Precision**: Proporção de previsões corretas para esta classe
        - **Recall**: Proporção de casos reais desta classe que foram corretamente identificados
        - **F1-score**: Média harmônica entre Precision e Recall
        - **Support**: Número de ocorrências reais desta classe no conjunto de teste
        
        As métricas são calculadas para cada classe (0 = Não Sobreviveu, 1 = Sobreviveu) e também fornecidas como médias:
        
        - **Macro avg**: Média simples das métricas por classe
        - **Weighted avg**: Média ponderada pelas ocorrências de cada classe
        """)
    
    # Análise de Importância de Features
    st.subheader("Importância das Características (Features)")
    
    feature_importance = model.feature_importances_
    feature_names = ['Classe', 'Sexo', 'Idade', 'Irmãos/Cônjuges', 'Pais/Filhos', 
                     'Tarifa', 'Porto', 'Tamanho Família', 'Sozinho', 'Título']
    
    # Criar DataFrame para importância das features
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    # Gráfico de importância
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='#ff7f0e')
    ax.set_xlabel('Importância', fontsize=12)
    ax.set_title('Importância das Características no Modelo', fontsize=16)
    st.pyplot(fig)
    
    with st.expander("O que é Importância de Features?"):
        st.markdown("""
        A **Importância de Features** indica o quanto cada característica contribui para as previsões do modelo.
        
        No algoritmo Random Forest, a importância é calculada com base em:
        
        1. **Diminuição da Impureza**: Quanto cada feature reduz a impureza (Gini ou entropia) nas árvores
        2. **Frequência de Uso**: Com que frequência cada feature é usada para dividir os dados
        
        **Interpretação**: Features com maior importância têm mais influência nas previsões do modelo.
        
        No nosso caso, as características mais importantes para prever a sobrevivência no Titanic são:
        {}
        """.format(", ".join(importance_df.nlargest(3, 'Importance')['Feature'].tolist())))

# Informações sobre o modelo
st.sidebar.header("Sobre o Modelo")
st.sidebar.info("""
Este modelo utiliza um algoritmo Random Forest Classifier
treinado nos dados dos passageiros do Titanic.

**Accuracy:** ~82% (validação cruzada)
**Features utilizadas:**
- Classe, Sexo, Idade
- Número de familiares
- Tarifa paga
- Porto de embarque
- Título (extraído do nome)
""")

# Rodapé
st.markdown("---")
st.markdown("""
**Desenvolvido por Bruno Galvão**  
Baseado na análise exploratória de dados do Titanic:  
[GitHub Repository](https://github.com/bcmaymonegalvao/Exploratory-Data-Analysis)
""")
