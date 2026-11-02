import tf from '@tensorflow/tfjs-node';

// criando a receita para o treinamento do modelo
async function trainModel(inputXs, outputYs) {
    
    // Configura um modelo de rede neural Feedfoward 
    const model = tf.sequential()

    // Primeira camada da rede:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizacoes)
    // O vetor de entrada tem que ser sempre o mesmo valor de features que tem

    // 80 neuronios
    // quanto mais neuronios, mais complexidade a rede pode aprender
    // e consequentemente, mais processamento ela vai usar

    // O método dense indica que é uma camada neural "totalmente conectada" (fully connected)
    // Cada neurónio presente nesta camada recebe dados de absolutamente todos os neurónios
    // da camada imediatamente anterior

    // A ReLU age como um filtro:
    // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    // Se a informação chegou nesse neuronio é positiva, passa para frente!
    // se for zero ou negativa, pode jogar fora, nao vai servir para nada
    // f(x) = max(0, x), muito utilizada nas camadas ocultas p ajudar no processamento
    model.add(tf.layers.dense({
        inputShape: [7], 
        units: 80,
        activation: 'relu'
    }))

    // Saída: 3 neuronios
    // um para cada categoria (premium, medium, basic)

    // activation: softmax normaliza a saida em probabilidades
    // Utilizada em problemas multiclasses ao inves do sigmoid
    model.add(tf.layers.dense({
        units: 3,
        activation: 'softmax'
    }))

    // optimizer: Define o algoritmo matemático encarregado de atualizar 
    // os pesos dos neurônios com base nos gradientes calculados.
    // "adam": É o otimizador padrão e mais seguro da indústria. 
    // Ele ajusta a taxa de aprendizado (learning rate) dinamicamente para cada parâmetro
    // durante o treino, garantindo uma convergência rápida. Ideal para problemas tabulares.
    
    //loss(Função de Custo / Perda): É a métrica de erro que o otimizador 
    // tentará minimizar agressivamente. Ela calcula a distância entre 
    // a previsão feita pela rede e a resposta real do dataset.
    // categoricalCrossentropy: Usada exclusivamente para classificação multiclasse 
    // onde os labels de saída estão em formato one-hot encoding

    // metrics: São indicadores calculados ao longo das épocas estritamente 
    // para avaliação humana e logs. Diferente da função loss, 
    // os resultados das métricas não são usados para atualizar os pesos da rede.

    // O earlystoping não é configurado no método `compile` e sim no `fit`
    model.compile({
        optimizer: "adam",
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })


    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0, 
            epochs: 100,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epochs, logs) => console.log(
                    `Epoch: ${epochs} - loss: ${logs.loss} - Acurácia: ${logs.acc}`
                )
            }
        }
    )

    return model
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores com one-hot encoded
// Ordem: [idade, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [30, 1, 0, 0, 1, 0, 0], // Erick
//     [25, 0, 1, 0, 0, 1, 0],    // Ana
//     [40, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const modelo = trainModel(inputXs, outputYs)