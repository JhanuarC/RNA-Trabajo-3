import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to import optional packages with graceful fallbacks
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    sns = None
    plt.style.use('ggplot')

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError:
    plot_acf = None
    plot_pacf = None

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'demanda_transporte_global.csv')
    output_dir = os.path.join(base_dir, 'plots_eda')
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(base_dir, 'eda_summary.txt')

    print("=========================================")
    print("  EJECUTANDO ANÁLISIS EXPLORATORIO (EDA) ")
    print("=========================================")
    print(f"Buscando archivo de datos en: {data_path}")

    if not os.path.exists(data_path):
        # Fallback to current working directory if run incorrectly
        data_path = 'demanda_transporte_global.csv'
        if not os.path.exists(data_path):
            print("Error: No se pudo encontrar demanda_transporte_global.csv")
            sys.exit(1)

    # 1. Cargar y Preparar Datos
    df = pd.read_csv(data_path)
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Extraer variables temporales
    df['mes'] = df['fecha'].dt.month
    df['dia_semana'] = df['fecha'].dt.weekday  # 0=Lunes, 6=Domingo
    df['es_fin_de_semana'] = df['dia_semana'].isin([4, 5]).astype(int)  # Viernes y Sábado según lógica de negocio original
    df['dia_mes'] = df['fecha'].dt.day

    # 2. Generar Reporte de Perfilado de Datos
    summary_lines = []
    def log_summary(text):
        print(text)
        summary_lines.append(text)

    log_summary("=========================================")
    log_summary("  PERFILADO DE DATOS (DATA PROFILING)    ")
    log_summary("=========================================")
    log_summary(f"Dimensiones del dataset: {df.shape[0]} registros, {df.shape[1]} columnas")
    log_summary("\nColumnas y tipos de datos:")
    for col, dtype in zip(df.columns, df.dtypes):
        log_summary(f" - {col}: {dtype}")

    missing = df.isnull().sum()
    log_summary("\nValores nulos por columna:")
    for col, count in missing.items():
        log_summary(f" - {col}: {count}")

    log_summary("\nEstadísticas descriptivas de demanda_pasajeros:")
    desc = df['demanda_pasajeros'].describe()
    for stat, val in desc.items():
        log_summary(f" - {stat}: {val:.2f}")

    log_summary("\nDistribución por hemisferio:")
    for hem, count in df['hemisferio'].value_counts().items():
        log_summary(f" - Hemisferio {hem}: {count} registros")

    log_summary(f"\nPaíses únicos ({df['pais'].nunique()}): {', '.join(sorted(df['pais'].unique()))}")
    log_summary(f"Tipos de transporte ({df['car_type'].nunique()}): {', '.join(df['car_type'].unique())}")
    log_summary(f"Métodos de pago ({df['payment_method'].nunique()}): {', '.join(df['payment_method'].unique())}")

    event_counts = df['evento_especial'].value_counts()
    log_summary("\nRegistros en eventos especiales:")
    log_summary(f" - Días regulares (0): {event_counts.get(0, 0)}")
    log_summary(f" - Días de eventos (1): {event_counts.get(1, 0)}")

    min_date = df['fecha'].min()
    max_date = df['fecha'].max()
    log_summary(f"\nRango temporal: desde {min_date.strftime('%Y-%m-%d')} hasta {max_date.strftime('%Y-%m-%d')}")

    # Guardar reporte de perfilado en archivo de texto
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nReporte de perfilado guardado exitosamente en: {summary_path}")

    # 3. Generar Visualizaciones

    # 3.1. Distribución de la variable objetivo (Pasajeros) y Boxplot
    print("\nGenerando Gráfico 1: Distribución de Demanda y Boxplot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if sns:
        sns.histplot(df['demanda_pasajeros'], kde=True, ax=axes[0], color='skyblue')
        sns.boxplot(y=df['demanda_pasajeros'], ax=axes[1], color='lightcoral')
    else:
        axes[0].hist(df['demanda_pasajeros'], bins=50, color='skyblue', edgecolor='black')
        axes[1].boxplot(df['demanda_pasajeros'])
    axes[0].set_title('Distribución de la Demanda de Pasajeros')
    axes[0].set_xlabel('Pasajeros')
    axes[0].set_ylabel('Frecuencia')
    axes[1].set_title('Detección de Outliers (Boxplot)')
    axes[1].set_ylabel('Pasajeros')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_demand_distribution.png'), dpi=150)
    plt.close()

    # 3.2. Tendencia Temporal Global Agregada
    print("Generando Gráfico 2: Tendencia Temporal Global...")
    daily_df = df.groupby('fecha')['demanda_pasajeros'].sum().reset_index()
    plt.figure(figsize=(14, 6))
    plt.plot(daily_df['fecha'], daily_df['demanda_pasajeros'], color='teal', alpha=0.5, label='Demanda Diaria Total')
    daily_df['rolling_avg_30d'] = daily_df['demanda_pasajeros'].rolling(window=30).mean()
    plt.plot(daily_df['fecha'], daily_df['rolling_avg_30d'], color='darkorange', linewidth=2.5, label='Media Móvil 30 días')
    plt.title('Evolución Temporal de la Demanda de Pasajeros (Total Agregado)')
    plt.xlabel('Fecha')
    plt.ylabel('Total Pasajeros Diarios')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_temporal_trend.png'), dpi=150)
    plt.close()

    # 3.3. Estacionalidad Mensual (Anual)
    print("Generando Gráfico 3: Estacionalidad Mensual...")
    monthly_avg = df.groupby('mes')['demanda_pasajeros'].mean().reset_index()
    months_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    monthly_avg['mes_nombre'] = monthly_avg['mes'].apply(lambda x: months_names[x-1])
    plt.figure(figsize=(10, 5))
    if sns:
        sns.barplot(data=monthly_avg, x='mes_nombre', y='demanda_pasajeros', palette='viridis', hue='mes_nombre', legend=False)
    else:
        plt.bar(monthly_avg['mes_nombre'], monthly_avg['demanda_pasajeros'], color='teal')
    plt.title('Demanda Promedio de Pasajeros por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Promedio de Pasajeros')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_monthly_seasonality.png'), dpi=150)
    plt.close()

    # 3.4. Estacionalidad Semanal (Día de la semana y Fin de semana)
    print("Generando Gráfico 4: Estacionalidad Semanal...")
    weekly_avg = df.groupby('dia_semana')['demanda_pasajeros'].mean().reset_index()
    days_names = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    weekly_avg['dia_nombre'] = weekly_avg['dia_semana'].apply(lambda x: days_names[x])

    weekend_avg = df.groupby('es_fin_de_semana')['demanda_pasajeros'].mean().reset_index()
    weekend_avg['tipo_dia'] = weekend_avg['es_fin_de_semana'].map({0: 'Día de Semana', 1: 'Fin de Semana (Vie-Sáb)'})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if sns:
        sns.barplot(data=weekly_avg, x='dia_nombre', y='demanda_pasajeros', ax=axes[0], palette='Blues_r', hue='dia_nombre', legend=False)
        sns.barplot(data=weekend_avg, x='tipo_dia', y='demanda_pasajeros', ax=axes[1], palette='Set2', hue='tipo_dia', legend=False)
    else:
        axes[0].bar(weekly_avg['dia_nombre'], weekly_avg['demanda_pasajeros'], color='royalblue')
        axes[1].bar(weekend_avg['tipo_dia'], weekend_avg['demanda_pasajeros'], color=['lightgray', 'seagreen'])
    axes[0].set_title('Demanda Promedio por Día de la Semana')
    axes[0].set_xlabel('Día de la Semana')
    axes[0].set_ylabel('Pasajeros')
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)
    axes[1].set_title('Diferencia de Demanda: Semana vs Fin de Semana')
    axes[1].set_xlabel('Tipo de Día')
    axes[1].set_ylabel('Pasajeros')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_weekly_seasonality.png'), dpi=150)
    plt.close()

    # 3.5. Comparativa de Variables Categóricas (País, Vehículo, Método de pago, Hemisferio)
    print("Generando Gráfico 5: Comparativa de Variables Categóricas...")
    country_avg = df.groupby('pais')['demanda_pasajeros'].mean().sort_values(ascending=False).reset_index()
    car_avg = df.groupby('car_type')['demanda_pasajeros'].mean().sort_values(ascending=False).reset_index()
    pay_avg = df.groupby('payment_method')['demanda_pasajeros'].mean().sort_values(ascending=False).reset_index()
    hem_avg = df.groupby('hemisferio')['demanda_pasajeros'].mean().sort_values(ascending=False).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    if sns:
        sns.barplot(data=country_avg, y='pais', x='demanda_pasajeros', ax=axes[0, 0], palette='coolwarm', hue='pais', legend=False)
        sns.barplot(data=car_avg, x='car_type', y='demanda_pasajeros', ax=axes[0, 1], palette='Set1', hue='car_type', legend=False)
        sns.barplot(data=pay_avg, x='payment_method', y='demanda_pasajeros', ax=axes[1, 0], palette='Set3', hue='payment_method', legend=False)
        sns.barplot(data=hem_avg, x='hemisferio', y='demanda_pasajeros', ax=axes[1, 1], palette='pastel', hue='hemisferio', legend=False)
    else:
        axes[0, 0].barh(country_avg['pais'], country_avg['demanda_pasajeros'], color='coral')
        axes[0, 1].bar(car_avg['car_type'], car_avg['demanda_pasajeros'], color='teal')
        axes[1, 0].bar(pay_avg['payment_method'], pay_avg['demanda_pasajeros'], color='purple')
        axes[1, 1].bar(hem_avg['hemisferio'], hem_avg['demanda_pasajeros'], color='green')

    axes[0, 0].set_title('Demanda Promedio por País')
    axes[0, 0].set_xlabel('Promedio de Pasajeros')
    axes[0, 0].set_ylabel('País')
    axes[0, 0].grid(True, axis='x', linestyle='--', alpha=0.5)

    axes[0, 1].set_title('Demanda Promedio por Tipo de Vehículo')
    axes[0, 1].set_xlabel('Tipo de Vehículo')
    axes[0, 1].set_ylabel('Pasajeros')
    axes[0, 1].grid(True, axis='y', linestyle='--', alpha=0.5)

    axes[1, 0].set_title('Demanda Promedio por Método de Pago')
    axes[1, 0].set_xlabel('Método de Pago')
    axes[1, 0].set_ylabel('Pasajeros')
    axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.5)

    axes[1, 1].set_title('Demanda Promedio por Hemisferio')
    axes[1, 1].set_xlabel('Hemisferio')
    axes[1, 1].set_ylabel('Pasajeros')
    axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_categorical_analysis.png'), dpi=150)
    plt.close()

    # 3.6. Impacto de Eventos Especiales (Anomalías)
    print("Generando Gráfico 6: Impacto de Eventos Especiales...")
    plt.figure(figsize=(10, 6))
    if sns:
        sns.boxplot(data=df, x='evento_especial', y='demanda_pasajeros', palette='Set2', hue='evento_especial', legend=False)
    else:
        groups = [df[df['evento_especial'] == 0]['demanda_pasajeros'], df[df['evento_especial'] == 1]['demanda_pasajeros']]
        plt.boxplot(groups, tick_labels=['Día Normal', 'Evento Especial'])
    plt.title('Impacto de Eventos Especiales en la Demanda de Pasajeros')
    plt.xlabel('Evento Especial (0 = No, 1 = Sí)')
    plt.ylabel('Demanda de Pasajeros')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_special_events_impact.png'), dpi=150)
    plt.close()

    # 3.7. Análisis de Autocorrelación (ACF/PACF) para una serie temporal individual
    # (Elegimos Japón - train_express - card como serie representativa)
    print("Generando Gráfico 7: Autocorrelación (ACF/PACF) - Caso de Estudio Japón...")
    df_rep = df[(df['pais'] == 'Japan') & (df['car_type'] == 'train_express') & (df['payment_method'] == 'card')].sort_values('fecha')
    series = df_rep['demanda_pasajeros'].values

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    if plot_acf and plot_pacf:
        plot_acf(series, lags=30, ax=axes[0], title='Autocorrelación (ACF) - Japón Express Train (Card)')
        plot_pacf(series, lags=30, ax=axes[1], title='Autocorrelación Parcial (PACF) - Japón Express Train (Card)')
    else:
        lags = range(1, 31)
        acf_vals = [pd.Series(series).autocorr(lag=l) for l in lags]
        axes[0].stem(lags, acf_vals)
        axes[0].set_title('Autocorrelación (ACF) - Japón Express Train (Card) [Cálculo Manual]')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('Correlación')
        axes[0].grid(True)
        
        axes[1].text(0.5, 0.5, 'Se requiere statsmodels instalado para calcular el PACF.', 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
        axes[1].set_title('Autocorrelación Parcial (PACF)')
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_acf_pacf.png'), dpi=150)
    plt.close()

    # 3.8. Matriz de Correlación
    print("Generando Gráfico 8: Matriz de Correlación...")
    # One-hot encoding para poder calcular correlaciones lineales
    df_encoded = pd.get_dummies(df, columns=['pais', 'hemisferio', 'car_type', 'payment_method'], drop_first=True)
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
    corr_matrix = df_encoded[numeric_cols].corr()

    # Seleccionar las top 15 características más correlacionadas con la demanda
    top_corr_features = corr_matrix['demanda_pasajeros'].abs().sort_values(ascending=False).index[:15]
    sub_corr_matrix = df_encoded[top_corr_features].corr()

    plt.figure(figsize=(12, 10))
    if sns:
        sns.heatmap(sub_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Matriz de Correlación de Pearson (Top 15 Características)', y=1.05)
        plt.tight_layout()
    else:
        # Fallback to matplotlib matshow
        fig = plt.gcf()
        ax = fig.add_subplot(111)
        cax = ax.matshow(sub_corr_matrix.values, cmap='coolwarm')
        fig.colorbar(cax)
        ax.set_xticks(range(len(top_corr_features)))
        ax.set_xticklabels(top_corr_features, rotation=90)
        ax.set_yticks(range(len(top_corr_features)))
        ax.set_yticklabels(top_corr_features)
        plt.title('Matriz de Correlación de Pearson (Top 15 Características)', y=1.15)
    
    plt.savefig(os.path.join(output_dir, '08_correlation_matrix.png'), dpi=150)
    plt.close()

    print("\n=========================================")
    print(f"  EDA COMPLETADO CON ÉXITO!               ")
    print(f"  Gráficos guardados en: {output_dir}")
    print("=========================================")

if __name__ == '__main__':
    main()
